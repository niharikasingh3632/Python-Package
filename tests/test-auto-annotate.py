from auto_annotate import AutoAnnotator 

def test_auto_annotate():
    input_folder = ""
    output_folder = ""
    model_name = ""
    classes_prompt = {}
    annotator = AutoAnnotator(model_name)
    annotator.run(input_folder, output_folder)
    assert annotator is not None

if __name__ == "__main__":
    test_auto_annotate()
