from src.classes.labelers.labeler import BDTOPOLabeler


def get_labeler(type_labeler, year, dep, task):
    labeler = None
    match type_labeler:
        case "BDTOPO":
            labeler = BDTOPOLabeler(year=year, dep=dep, task=task)
        case _:
            pass
    return labeler
