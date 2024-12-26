from typing import Dict, Any

class GuidanceHandler:
    def __init__(self):
        pass

    def create_program(self, template: str) -> str:
        """Create a program template"""
        return template

    def execute_program(
        self,
        program: str,
        variables: Dict[str, Any]
    ) -> str:
        try:
            # Replace variables in the template
            template = program
            for key, value in variables.items():
                template = template.replace(f"{{{{{key}}}}}", str(value))
            return f"Mock execution: {template}"
        except Exception as e:
            raise Exception(f"Generation error: {str(e)}")