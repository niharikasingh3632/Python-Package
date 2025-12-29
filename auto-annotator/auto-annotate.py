from grounding_dino import GroundingDINORun


available_models = ["grounding_dino"]

class AutoAnnotator:
    def __init__(self, model_name):
        self.model_name = model_name

        if self.model_name not in available_models:
            raise ValueError(f"Model {self.model_name} is not available. Available models: {available_models}")

    def run(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder

        if self.model_name == "grounding_dino":
            try:
                runner = GroundingDINORun(self.input_folder, self.output_folder)
                runner.run()
            except Exception as e:
                print(f"Error running GroundingDINORun: {e}")




        
