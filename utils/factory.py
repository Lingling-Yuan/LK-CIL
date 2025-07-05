def get_model(model_name, args):
    name = model_name.lower()
    if name == "lkcil":
        from models.lkcil import Learner
    else:
        raise NotImplementedError(f"Unknown model: {model_name}")
    return Learner(args)
