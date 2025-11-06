import ast
from pathlib import Path
from .Command import Command


class InitCommand(Command):
    @staticmethod
    def run():
        path = Path(".")
        init_file = path / "__init__.py"
        init_file.write_text("")
        for file in path.iterdir():
            if file.name == "__init__.py" or file.suffix != ".py":
                continue
            tree = ast.parse(file.read_text())
            classes = [
                node.name
                for node in tree.body
                if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
            ]
            functions = [
                node.name
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
            ]
            if not (classes or functions):
                continue
            module = file.stem
            lines = [
                f"from .{module} import {name}\n" for name in (*classes, *functions)
            ]
            with init_file.open("a") as f:
                f.writelines(lines)
            print(f"Processed {file}: classes={classes}, functions={functions}")
