from pathlib import Path
from .Namespace import Namespace


class MakeNamespace(Namespace):
    def trait(self, name: str):
        class_name = name.split(".")[-1]
        file = name.replace(".", "/") + ".py"
        path = Path(file)
        parent = path.parent
        parent.mkdir(parents=True, exist_ok=True)
        path.touch()

        # copy the tempalte path
        current_path = Path(__file__).parent.parent.resolve()
        template_path = current_path / "templates" / "trait.txt"
        content = template_path.read_text()
        content = content.replace("{{CLASS}}", class_name)
        path.write_text(content)
