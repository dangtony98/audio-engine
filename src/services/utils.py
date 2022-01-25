from datetime import date


def compute_age(birthdate):
    """
    Compute age given the person's birthdate
    """
    try:
        today = date.today()
        age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
    except:
        age = 0
    return age