# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------

from pycuda.compiler import SourceModule
from pycuda.tools import context_dependent_memoize
from pytools import memoize_method
import pycuda.driver as drv
import flexpt_array

_common_template = r"""

#define __dev__ __device__ __forceinline__

__dev__ float fscale_from_flex(int iwl)
{
    float ret;
    int exponent = (127 - (15 - iwl)) << 23;
    asm("mov.b32 %0, %1;" : "=f"(ret) : "r"(exponent));
    return ret;
}
__dev__ float fscale_to_flex(int iwl)
{
    float ret;
    int exponent = (127 + (15 - iwl)) << 23;
    asm("mov.b32 %0, %1;" : "=f"(ret) : "r"(exponent));
    return ret;
}
__dev__ float flex_to_float(short val, float fscale)
{
    float ret;
    asm("cvt.rp.f32.s16 %0, %1;\n\t"
        "mul.rp.ftz.f32 %0, %0, %2;"
        : "=f"(ret) : "h"(val), "f"(fscale));
    return ret;
}
__dev__ short float_to_flex(float val, float fscale)
{
    short ret;
    asm ("mul.rp.ftz.f32 %1, %1, %2;\n\t"
         "cvt.rpi.sat.s16.f32 %0, %1;"
         : "=h"(ret) : "f"(val), "f"(fscale));
    return ret;
}
__dev__ short int_to_flex(int val)
{
    short ret;
    asm("cvt.sat.s16.s32 %0, %1;" : "=h"(ret) : "r"(val));
    return ret;
}
__dev__ int flex_to_int(short val)
{
    int ret;
    asm ("cvt.s32.s16 %0, %1;" : "=r"(ret) : "h"(val));
    return ret;
}

"""

_ew_template = r"""

%(common)s

#define UNROLL_SHIFT %(unroll_factor)s
#define UNROLL_COUNT (1 << UNROLL_SHIFT)

__global__ void %(name)s (
    short *out, const int ld_out, const int step_out, const int n, 
    %(arguments)s)
{
    const int tid = threadIdx.x;
    const int bx  = blockIdx.x;
    const int by  = blockIdx.y;
    const int numThreads = blockDim.x;
    const int numBlocks  = gridDim.y;

    out += bx * ld_out;
    %(offsets)s

    %(scales)s

    const int n2 = (n+(UNROLL_COUNT-1)) >> UNROLL_SHIFT;

    for (int i = by * numThreads + tid; i < n2; i += numThreads * numBlocks)
    {
        int i2 = i << UNROLL_SHIFT;

        %(declares)s

        #pragma unroll
        for (int j = 0; j < UNROLL_COUNT; j++)
        {
            %(loads)s
        }

        #pragma unroll
        for (int j = 0; j < UNROLL_COUNT; j++)
        {
            %(operations)s

            if (j == 0 || i2 + j < n)
                out[step_out * (i2 + j)] = %(output)s
        }        
    }
}
""" 

_ew_strings = {

    "array" : {
        "arguments" : "const short* a_in{0}, const int ld_in{0}, const int step_in{0}, " + \
                      "const short iwl_in{0}, const short flags{0}",
        "offsets"   : "if ((flags{0} & 1) == 0) a_in{0} += bx * ld_in{0};",
        "scales"    : "float fscale_a{0} = fscale_from_flex(iwl_in{0});",
        "declares"  : "short data{0}[UNROLL_COUNT];",
        "loads"     : "data{0}[j] = __ldg(flags{0} & 2 ? a_in{0} : a_in{0} + step_in{0} * (i2 + j));",
        "operations": "float a{0} = flex_to_float(data{0}[j], fscale_a{0});",
    },
    "const" : {
        "arguments" : "short c_in{0}, short iwl_c{0}",
        "scales"    : "float c{0} = flex_to_float(c_in{0}, fscale_from_flex(iwl_c{0}));",
    },
    "op" : {
        "arguments" : "short iwl_op{0}",
        "scales"    : "float fscale_op_to{0} = fscale_to_flex(iwl_op{0}); " + \
                      "float fscale_op_from{0} = fscale_from_flex(iwl_op{0});",
        "operations": "r{0} = flex_to_float(float_to_flex(r{0}, fscale_op_to{0}), fscale_op_from{0});"
    },
    "output" : {
        "output"    : "float_to_flex(r{0}, fscale_op_to{0});"
    },
}

# Note: binary operands come off the stack in reverse order
_float_operations = {

    "add"  : (2, 'float {0} = __fadd_ru({2}, {1});' ),
    "sub"  : (2, 'float {0} = __fsub_ru({2}, {1});' ),
    "mul"  : (2, 'float {0} = __fmul_ru({2}, {1});' ),
    "div"  : (2, 'float {0} = __fdividef({2}, {1});' ),
    "copy" : (1, "float {0} = {1};"          ),
    "eq"   : (2, "float {0} = {2} == {1};"   ),
    "ne"   : (2, "float {0} = {2} != {1};"   ),
    "lt"   : (2, "float {0} = {2} <  {1};"   ),
    "le"   : (2, "float {0} = {2} <= {1};"   ),
    "gt"   : (2, "float {0} = {2} >  {1};"   ),
    "ge"   : (2, "float {0} = {2} >= {1};"   ),
    "min"  : (2, "float {0} = min({2},{1});" ),
    "max"  : (2, "float {0} = max({2},{1});" ),
    "neg"  : (1, "float {0} = -{1};"         ),
    "abs"  : (1, "float {0} = abs({1});"     ),
    "sqrt" : (1, "float {0} = sqrtf({1});"   ),
    "sqr"  : (1, "float {0} = {1} * {1};"    ),
    "pow"  : (1, "float {0} = powf({1});"    ),
    "exp"  : (1, "float {0} = expf({1});"    ),
    "log"  : (1, "float {0} = logf({1});"    ),
    "exp2" : (1, "float {0} = exp2f({1});"   ),
    "log2" : (1, "float {0} = log2f({1});"   ),
    "sig"  : (1, "float {0} = 1.0f/(1.0f + expf(-{1}));"),
    "sig2" : (1, "float {0} = 1.0f/(1.0f + exp2f(-{1}));"),
    "tanh" : (1, "float {0} = tanhf({1});"   ),
    "tanh2": (1, "float {0} = (exp2f(2.0f*{1}) - 1.0f) / (exp2f(2.0f*{1}) + 1.0f);" ),
}

_int_operations = {
    "not"  : (1, "int {0} = ~{1};"       ),
    "and"  : (2, "int {0} = {2} &  {1};" ),
    "or"   : (2, "int {0} = {2} |  {1};" ),
    "xor"  : (2, "int {0} = {2} ^  {1};" ),
    "shl"  : (2, "int {0} = {2} << {1};" ),
    "shr"  : (2, "int {0} = {2} >> {1};" ),
    "l_shr": (2, 'int {0}; asm("shr.u32 %0, %1, %2;":"=r"({0}):"r"({2}),"r"({1}));'),
}

def _get_module(template, template_vals):

    code = template % template_vals

    f = open("%s.cu" % template_vals["name"], "w")
    print >>f, code
    f.close()

    return SourceModule(code, options=["--use_fast_math" ], keep=False) #,"-G"


@context_dependent_memoize
def _get_compound_ew_kernel(unroll_factor, type_args):

    sig       = "Piii"
    stack     = []
    array_ids = set()
    
    template_vals = { 
        "name"          : "kernel_",
        "unroll_factor" : unroll_factor,
        "common"        : _common_template
    }

    for key in ("arguments","offsets","scales","declares","loads","operations","output"):
        template_vals[key] = []

    for arg_i, arg in enumerate(type_args):

        arg_type, arg_idx = arg[0:2]

        # Array operands
        if arg_type is flexpt_array.FlexptArray:

            template_vals["name"] += chr(ord("A") + arg_idx)
            stack.append("a%d" % arg_idx)

            if arg_idx not in array_ids:
                
                array_ids.add(arg_idx)

                sig += "Piihh"
                for key in ("arguments","offsets","scales","declares","loads","operations"):
                    template_vals[key].append(_ew_strings["array"][key].format(arg_idx))

        # Constant operands
        elif arg_type is int:

            sig  += "hh"
            template_vals["name"] += chr(ord("a") + arg_idx)
            stack.append("c%d" % arg_idx)
            for key in ("arguments","scales"):
                template_vals[key].append(_ew_strings["const"][key].format(arg_idx))

        # Operations
        else:
            op_name, op_idx, postop_convert_to_flex = arg

            template_vals["name"] += "_%s_" % op_name

            (num_ops, op_code) = _float_operations[op_name]

            op_list = [ "r%d" % op_idx ]

            #build the operands from the stack
            for i in range(num_ops):
                op_list.append(stack.pop())

            template_vals["operations"].append(op_code.format(*op_list))

            stack.append(op_list[0])

            if postop_convert_to_flex: 
                sig += "h"
                if arg_i+1 < len(type_args):
                    for key in ("arguments","scales","operations"):
                        template_vals[key].append(_ew_strings["op"][key].format(op_idx))
                else:
                    for key in ("arguments","scales"):
                        template_vals[key].append(_ew_strings["op"][key].format(op_idx))

    for key in ("output",):
        template_vals[key].append(_ew_strings["output"][key].format(op_idx))

    template_vals["arguments"]  = ", ".join(template_vals["arguments"])
    template_vals["offsets"]    = "\n    ".join(template_vals["offsets"])
    template_vals["scales"]     = "\n    ".join(template_vals["scales"])
    template_vals["declares"]   = "\n        ".join(template_vals["declares"])
    template_vals["loads"]      = "\n            ".join(template_vals["loads"])
    template_vals["operations"] = "\n            ".join(template_vals["operations"])
    template_vals["output"]     = "".join(template_vals["output"])
    template_vals["name"]      += str(unroll_factor)

    module = _get_module(_ew_template, template_vals)
    kernel = module.get_function(template_vals["name"])
    kernel.prepare(sig)

    return kernel


def call_compound_ew_kernel(out, *args):
    """
    Pass in a list of FlexptArray objects, constants (python int or float) and operators (str, iwl).
    The first item should be the output FlexptArray to store the results.
    The rest of the arguments should be in postfix notation.
    Operators are tuples with the name as the first item and 
    the resulting integer word length (iwl) as the second.
    If iwl is None it will be infered from the output assignment iwl value.
    The final operator should have an iwl of None or should match the iwl of the result array.


    C +=  2.5 * A * B + 1
    call_compound_ew_kernel(C, 2.5, A, ("mul",2), B, ("mul",4), 1, ("add", 4), C, "add")
    """

    arg_cnt     = 0
    op_cnt      = 0
    array_ids   = {}
    kernel_args = []
    type_args   = []
    tune_args   = []

    if isinstance(out, flexpt_array.FlexptArray):
        kernel_args.extend((out.gpudata, out.strides[0]//2, out.strides[1]//2, out.shape[1]))
    else:
        raise TypeError("First arg must be instance of FlexptArray for output")

    if out.is_trans:
        raise ValueError("EW kernels don't yet work on transposed views")

    # if not specified the last iwl takes the value from the output array
    # if it is specified the output iwl is modified 
    if args[-1].get("iwl") is None:
        args[-1]["iwl"] = out.iwl

    for arg in args:
        
        # Array operand
        if isinstance(arg, flexpt_array.FlexptArray):

            # If same array is passed in multiple times to expression,
            # consolidate them into one kernel argument.
            if arg in array_ids:
                indx = array_ids[arg]
            else:

                if arg.is_trans:
                    raise ValueError("EW kernels don't yet work on transposed views")

                if  arg.shape[0] != out.shape[0] and arg.shape[0] != 1 or \
                    arg.shape[1] != out.shape[1] and arg.shape[1] != 1:
                    raise ValueError(
                        "Input shape:%s not compatible with output:%s" % (
                        str(arg.shape), str(out.shape)))

                indx = array_ids[arg] = arg_cnt
                arg_cnt += 1

                # support broadcast
                flags = 0
                if arg.shape[0] == 1: flags |= 1
                if arg.shape[1] == 1: flags |= 2

                kernel_args.extend((arg.gpudata, arg.strides[0]//2, arg.strides[1]//2, arg.iwl, flags))

                tune_args.append(flags)

            type_args.append((flexpt_array.FlexptArray, indx))

        # Constant operand
        elif type(arg) in (int, float):
            
            type_args.append((int, arg_cnt))
            kernel_args.extend(flexpt_array.Flexpt.flex_from_native(arg))
            arg_cnt += 1

        # Operation 
        elif type(arg) is dict:

            op_name = arg["op"]

            if op_name not in _float_operations:
                raise ValueError("%s is not a valid operation" % op_name)

            if arg.get("iwl") is None: 
                type_args.append((op_name, op_cnt, 0))
            else:
                type_args.append((op_name, op_cnt, 1))
                kernel_args.append(arg["iwl"])
            
            op_cnt += 1
            
        else:
            raise TypeError("args must be instance of FlexptArray, int, float, (str, int) or str")

    n, k = out.shape

    #TODO: ceiling n and k to nearest 32?
    # n = (n // 32 + (n % 32 != 0)) * 32

    # make hashable for memoize
    type_args =tuple(type_args)
    tune_args =tuple(tune_args)

    (unroll_factor, blocks, threads) = (0, 1, 32) #_autotune_kernel(n, k, type_args, tune_args)

    kernel = _get_compound_ew_kernel(unroll_factor, type_args)

    kernel.prepared_call((n,blocks,1), (threads,1,1), *kernel_args)

    # only set this at the end as out can also be an input
    out.iwl = args[-1]["iwl"]

    return out

_context_warmup_set = set()

@context_dependent_memoize
def _autotune_kernel(n, k, type_args, tune_args):

    load_cnt    = 0
    array_ids   = set()
    kernel_args = list()
    tune_args   = list(tune_args)

    # Setup some fake data to autotune on
    # Perhaps tune on the real data but we'd need a custom memoize for that,
    # And we wouldn't be able to use n,k abstracted sizing
    data = drv.mem_alloc(n * k * 2)
    drv.memset_d16(data, 0, n * k)
    kernel_args.extend((data, k, 1, k))

    for arg in type_args:

        arg_type, arg_idx = arg[0:2]

        # Array operands
        if arg_type is flexpt_array.FlexptArray:
            if arg_idx not in array_ids:
                array_ids.add(arg_idx)
                flags = tune_args.pop()

                size = 1
                if flags & 1 == 0:
                    size *= n
                if flags & 2 == 0:
                    size *= k

                data = drv.mem_alloc(size * 2)
                drv.memset_d16(data, 0, size)
                kernel_args.extend((data, k, 1, 15, flags))
                load_cnt += 1

        # Constant operands
        elif arg_type is int:

            kernel_args.extend((0,15))

        # Operations
        elif arg[2]: # postop_convert_to_flex

            kernel_args.append(15)

    repeat = min(500, max(1, 8192**2 // (n * k)))

    # warm up the gpu clocks so our timings are accurate
    cur_ctx = drv.Context.get_current()
    if cur_ctx not in _context_warmup_set:
        _context_warmup_set.add(cur_ctx)
        kernel = _get_compound_ew_kernel(0, type_args)
        for r in range(repeat * 10):
            kernel.prepared_call((n,1,1), (32,1,1), *kernel_args)


    start = drv.Event()
    end   = drv.Event()

    min_msecs   = 99999999.9
    min_blocks  = 1
    min_threads = 32
    min_factor  = 0
    max_factor  = 3 if load_cnt < 4 else 2

    for unroll_factor in range(max_factor):

        kernel = _get_compound_ew_kernel(unroll_factor, type_args)
        unroll = 1 << unroll_factor

        for threads in (32,64,128,256):

            for blocks in (1,2,4,8,16,32,64,128,256,512,1024):

                loads = blocks * threads * unroll

                if loads > k and min_msecs != 99999999.9: 
                    #print "skipping %d loads for %d" % (loads, k)
                    continue

                loops = k // loads

                if (loops > 8 or loops < 1) and min_msecs != 99999999.9: 
                    print "skipping %d loops %d // %d" % (loops, k, loads) 
                    continue

                start.record()
                for r in range(repeat):
                    kernel.prepared_call((n,blocks,1), (threads,1,1), *kernel_args)
                end.record()
                end.synchronize()
                msecs = end.time_since(start)

                if msecs < min_msecs:
                    min_msecs   = msecs
                    min_factor  = unroll_factor
                    min_blocks  = blocks
                    min_threads = threads

                #print "%4d %d %4d %3d %4d (%4d, %4d) %4d %.5f" % \
                #    (repeat, unroll, blocks, threads, loads, n, k, loops, msecs)

    #print "Final: %d %4d %3d %.5f" % (1<<min_factor, min_blocks, min_threads, min_msecs)

    return (min_factor, min_blocks, min_threads)
