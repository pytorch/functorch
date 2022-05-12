import torch
import torch.fx as fx

from torch.fx.node import Node, Argument, Target, map_arg, _type_repr, _get_qualified_name
from torch.fx.graph import _is_from_torch, _Namespace, _origin_type_map, inplace_methods, _format_target, _custom_builtins

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Any, List, Dict, NamedTuple, Optional, Tuple, Set, FrozenSet, Type

reflectable_magic_methods = {
    'add': '{} + {}',
    'sub': '{} - {}',
    'mul': '{} * {}',
    'floordiv': '{} // {}',
    'truediv': '{} / {}',
    'div': '{} / {}',
    'mod': '{} % {}',
    'lshift': '{} << {}',
    'rshift': '{} >> {}',
    'and_': '{} & {}',
    'or_': '{} | {}',
    'xor': '{} ^ {}',
}

magic_methods = dict({
    'eq': '{} == {}',
    'ne': '{} != {}',
    'lt': '{} < {}',
    'gt': '{} > {}',
    'le': '{} <= {}',
    'ge': '{} >= {}',
    'pos': '+{}',
    'neg': '-{}',
    'invert': '~{}'}, **reflectable_magic_methods)


@dataclass
class JiteratorCode:
    """
    Represents all the information necessary to exec or save a graph as Python code.
    """
    # Python source code for the forward function definition.
    src: str
    # Values in global scope during exection of `src_def`.
    globals: Dict[str, Any]


sigmoid_code_string = '''
template <typename T> T _sigmoid(T x) {
    return T{1} / (T{1} + std::exp(-x));
}
'''

where_code_string = '''
template <typename T> T _where(bool cond, T a, T b){
    return cond ? a : b;
}
'''

# complex support missing
reciprocal_code_string = '''
template <typename scalar_t>
__device__ static inline scalar_t _reciprocal(scalar_t a) {
    return static_cast<scalar_t>(1)/a;
}
'''

# TODO: consider other impl
sign_code_string = '''
template <typename scalar_t>
static __device__ scalar_t _sign(scalar_t val) {
    return (0 < val) - (val < 0);
}
'''

# isneginf_code_string = '''
# template <typename scalar_t>
# static __device__ bool _isneginf(scalar_t a) {
#     return a == -::numeric_limits<scalar_t>::infinity();
# }
# '''

# isposinf_code_string = '''
# template <typename scalar_t>
# static __device__ bool _isposinf(scalar_t a) {
#     return a == std::numeric_limits<scalar_t>::infinity();
# }
# '''

class JiteratorCodeGen(object):
    def __init__(self):
        pass

    def get_predefined_kernel(self, func: str) -> str:
        predefined_kernel_map = {
            "sigmoid" : sigmoid_code_string,
            "where": where_code_string,
            "reciprocal": reciprocal_code_string,
            "sign": sign_code_string,
            # "isneginf": isneginf_code_string,
            # "isposinf": isposinf_code_string,
        }

        if func in predefined_kernel_map:
            return predefined_kernel_map[func]
        else:
            return None


    def gen_fn_def(self, free_vars: List[str]) -> str:
        """
        Given the free variables, generates the beginning of the FX function.
        By default, `gen_fn_def(['a', 'b'], '') == 'template <typename T> T my_kernel(T a, T b)'`
        """
        typed_vars = [f"T {var}" for var in free_vars]
        return f"template <typename T> T my_kernel({', '.join(typed_vars)})"

    def generate_output(self, output_args: Argument) -> str:
        """
        Given the output arguments, generates the return statement of the FX function.
        Note: The returned statement should not be indented.
        """

        if type(output_args) is torch.fx.immutable_collections.immutable_list:
            # assert len(output_args) == 1
            return f'return {repr(output_args[0])}'

        return f'return {repr(output_args)}'

    def _gen_jiterator_code(self, nodes, root_module: str) -> JiteratorCode:
        free_vars: List[str] = []
        body: List[str] = []

        # TODO: add kernel name here, not body
        inline_kernels: Set[str] = set()
        globals_: Dict[str, Any] = {}

        namespace = _Namespace()

        def add_global(name_hint: str, obj: Any):
            """Add an obj to be tracked as a global.

            We call this for names that reference objects external to the
            Graph, like functions or types.

            Returns: the global name that should be used to reference 'obj' in generated source.
            """
            if _is_from_torch(obj) and obj != torch.device:  # to support registering torch.device
                # HACK: workaround for how torch custom ops are registered. We
                # can't import them like normal modules so they must retain their
                # fully qualified name.
                return _get_qualified_name(obj)

            # normalize the name hint to get a proper identifier
            global_name = namespace.create_name(name_hint, obj)

            if global_name in globals_:
                assert globals_[global_name] is obj
                return global_name
            globals_[global_name] = obj
            return global_name

        # Pre-fill the globals table with registered builtins.
        for name, (_, obj) in _custom_builtins.items():
            add_global(name, obj)

        def type_repr(o : Any):
            if o == ():
                # Empty tuple is used for empty tuple type annotation Tuple[()]
                return '()'

            typename = _type_repr(o)

            if hasattr(o, '__origin__'):
                # This is a generic type, e.g. typing.List[torch.Tensor]
                origin_type = _origin_type_map.get(o.__origin__, o.__origin__)
                origin_typename = add_global(_type_repr(origin_type), origin_type)

                if hasattr(o, '__args__'):
                    # Assign global names for each of the inner type variables.
                    args = [type_repr(arg) for arg in o.__args__]

                    if len(args) == 0:
                        # Bare type, such as `typing.Tuple` with no subscript
                        # This code-path used in Python < 3.9
                        return origin_typename

                    return f'{origin_typename}[{",".join(args)}]'
                else:
                    # Bare type, such as `typing.Tuple` with no subscript
                    # This code-path used in Python 3.9+
                    return origin_typename

            # Common case: this is a regular module name like 'foo.bar.baz'
            return add_global(typename, o)

        def _format_args(args: Tuple[Argument, ...], kwargs: Dict[str, Argument]) -> str:
            def _get_repr(arg):
                # Handle NamedTuples (if it has `_fields`) via add_global.
                if isinstance(arg, tuple) and hasattr(arg, '_fields'):
                    qualified_name = _get_qualified_name(type(arg))
                    global_name = add_global(qualified_name, type(arg))
                    return f"{global_name}{repr(tuple(arg))}"
                return repr(arg)
            args_s = ', '.join(_get_repr(a) for a in args)
            kwargs_s = ', '.join(f'{k} = {_get_repr(v)}' for k, v in kwargs.items())
            if args_s and kwargs_s:
                return f'{args_s}, {kwargs_s}'
            return args_s or kwargs_s

        # Run through reverse nodes and record the first instance of a use
        # of a given node. This represents the *last* use of the node in the
        # execution order of the program, which we will use to free unused
        # values
        node_to_last_use : Dict[Node, Node] = {}
        user_to_last_uses : Dict[Node, List[Node]] = {}

        def register_last_uses(n : Node, user : Node):
            if n not in node_to_last_use:
                node_to_last_use[n] = user
                user_to_last_uses.setdefault(user, []).append(n)

        for node in reversed(nodes):
            map_arg(node.args, lambda n: register_last_uses(n, node))
            map_arg(node.kwargs, lambda n: register_last_uses(n, node))

        def get_cuda_function_name(func: str) -> str:
            kernel_name_map = {
                "signbit": "::signbit",
                "isfinite": "::isfinite",
                "isnan": "::isnan",
                "abs": "::abs",
                "acos": "::acos",
                "asin": "::asin",
                "atan": "::atan",
                "atan2": "::atan2",
                "ceil": "::ceil",
                "cos": "::cos",
                "cosh": "::cosh",
                "exp": "::exp",
                "fabs": "::fabs",
                "floor": "::floor",
                "fmod": "::fmod",
                "frexp": "::frexp",
                "ldexp": "::ldexp",
                "log": "::log",
                "log10": "::log10",
                "pow": "::pow",
                "sin": "::sin",
                "sinh": "::sinh",
                "sqrt": "::sqrt",
                "tan": "::tan",
                "tanh": "::tanh",
                "acosh": "::acosh",
                "asinh": "::asinh",
                "atanh": "::atanh",
                "cbrt": "::cbrt",
                "copysign": "::copysign",
                "erf": "::erf",
                "erfc": "::erfc",
                "exp2": "::exp2",
                "expm1": "::expm1",
                "fdim": "::fdim",
                "fma": "::fma",
                "fmax": "::fmax",
                "fmin": "::fmin",
                "hypot": "::hypot",
                "ilogb": "::ilogb",
                "lgamma": "::lgamma",
                "llrint": "::llrint",
                "llround": "::llround",
                "log1p": "::log1p",
                "log2": "::log2",
                "logb": "::logb",
                "lrint": "::lrint",
                "lround": "::lround",
                "nan": "::nan",
                "nearbyint": "::nearbyint",
                "nextafter": "::nextafter",
                "remainder": "::remainder",
                "remquo": "::remquo",
                "rint": "::rint",
                "round": "::round",
                "scalbln": "::scalbln",
                "scalbn": "::scalbn",
                "tgamma": "::tgamma",
                "trunc": "::trunc",
                "truncf": "::truncf",

                "rsqrt": "::rsqrt",  # doesn't support complex type
                "erfinv": "::erfinv",
                "isinf": "::isinf",

                "maximum": "::max",
                "minimum": "::min",
            }

            if func in kernel_name_map:
                return kernel_name_map[func]
            else:
                return None

        def emit_node(node : Node):
            if node.op == 'placeholder':
                assert isinstance(node.target, str)
                free_vars.append(f'{node.target}')
                raw_name = node.target.replace('*', '')
                if raw_name != repr(node):
                    body.append(f'auto {repr(node)} = {raw_name}\n')
                return
            elif node.op == 'call_method':
                assert isinstance(node.target, str)
                if node.target in magic_methods:
                    assert isinstance(node.args, tuple)
                    body.append(f'auto {repr(node)} = '
                                f'{magic_methods[node.target].format(*(repr(a) for a in node.args))}')
                    return

                # TODO: handle inplace methods, e.g. add_

                cuda_function_name = get_cuda_function_name(node.target)
                body.append(f'auto {repr(node)} = {cuda_function_name}({_format_args(node.args, node.kwargs)})')
                return
            elif node.op == 'call_function':
                assert callable(node.target)
                # pretty print operators
                if node.target.__module__ in ['_operator', 'torch', 'torch._ops.aten'] and node.target.__name__ in magic_methods:
                    assert isinstance(node.args, tuple)
                    body.append(f'auto {repr(node)} = '
                                f'{magic_methods[node.target.__name__].format(*(repr(a) for a in node.args))}')
                    return

                # pretty print inplace operators; required for jit.script to work properly
                # not currently supported in normal FX graphs, but generated by torchdynamo
                if node.target.__module__ == '_operator' and node.target.__name__ in inplace_methods:
                    body.append(f'{inplace_methods[node.target.__name__].format(*(repr(a) for a in node.args))};  '
                                f'auto {repr(node)} = {repr(node.args[0])}')
                    return


                cuda_function_name = get_cuda_function_name(node.target.__name__)
                if cuda_function_name:
                    body.append(f'auto {repr(node)} = {cuda_function_name}({_format_args(node.args, node.kwargs)})')
                    return


                predefined_code_str = self.get_predefined_kernel(node.target.__name__)
                if predefined_code_str:
                    body.append(f'auto {repr(node)} = _{node.target.__name__}<T>({_format_args(node.args, node.kwargs)})')
                    inline_kernels.add(predefined_code_str)
                    return

                assert False, f"not supported case found for '{node.target.__module__}.{node.target.__name__}'"
                # global_name = add_global(qualified_name, node.target)

                # print ("global_name ", global_name)

                # special case for getattr: node.args could be 2-argument or 3-argument
                # 2-argument: attribute access; 3-argument: fall through to attrib function call with default value
                # if global_name == 'getattr' and \
                #    isinstance(node.args, tuple) and \
                #    isinstance(node.args[1], str) and \
                #    node.args[1].isidentifier() and \
                #    len(node.args) == 2:
                #     body.append(f'auto {repr(node)} = {_format_target(repr(node.args[0]), node.args[1])}')
                #     return

            elif node.op == 'call_module':
                assert isinstance(node.target, str)
                body.append(f'auto {repr(node)} = '
                            f'{_format_target(root_module, node.target)}({_format_args(node.args, node.kwargs)})')
                return
            elif node.op == 'get_attr':
                assert isinstance(node.target, str)
                free_vars.append(f'{node.target}')
                # body.append(f'auto {repr(node)} = {_format_target(root_module, node.target)}')
                return
            elif node.op == 'output':
                body.append(self.generate_output(node.args[0]))
                return
            raise NotImplementedError(f'node: {node.op} {node.target}')

        for node in nodes:
            emit_node(node)
            body.append(';\n')

        if len(body) == 0:
            # If the Graph has no non-placeholder nodes, no lines for the body
            # have been emitted. To continue to have valid Python code, emit a
            # single pass statement
            body.append(';\n')

        inline_code = ''.join(inline_kernels)
        inline_code = '\n'.join('    ' + line for line in inline_code.split('\n'))

        prologue = self.gen_fn_def(free_vars)

        code = ''.join(body)
        code = '\n'.join('    ' + line for line in code.split('\n'))
        fn_code = f"""
{inline_code}
{prologue} {{
{code}
}}
"""
        return JiteratorCode(fn_code, globals_)
