import textwrap
import jinja2
import jinja2.meta


class Template:
    def __init__(
            self,
            source: str,
            rstrip: bool = True,
            dedent: bool = True,
            trim_blocks: bool = True,
            lstrip_blocks: bool = True,
            keep_trailing_newline: bool = False,
            autoescape: bool = False,
    ):
        self.source = source
        if rstrip:
            self.source = '\n'.join(line.rstrip() for line in self.source.splitlines())
            if source.endswith('\n'):
                self.source += '\n'
        if dedent:
            self.source = textwrap.dedent(self.source)
        self.template = jinja2.Template(
            source=self.source,
            trim_blocks=trim_blocks,
            lstrip_blocks=lstrip_blocks,
            keep_trailing_newline=keep_trailing_newline,
            autoescape=autoescape,
        )

    def __call__(self, *args, **kwargs) -> str:
        variables = dict(*args, **kwargs)
        text = self.template.render(variables)
        return text

    @property
    def input_variables(self) -> set[str]:
        ast = self.template.environment.parse(self.source)
        return jinja2.meta.find_undeclared_variables(ast)
