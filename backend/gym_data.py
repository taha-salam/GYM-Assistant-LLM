# Placeholder gym data. Edit freely to make it feel like a real gym.

CLASS_SCHEDULE = [
    {"day": "Monday", "time": "7:00 AM", "class": "Yoga", "instructor": "Taha "},
    {"day": "Monday", "time": "6:00 PM", "class": "HIIT", "instructor": "Adil"},
    {"day": "Tuesday", "time": "8:00 AM", "class": "Spin", "instructor": "Immad"},
    {"day": "Tuesday", "time": "7:00 PM", "class": "Strength Training", "instructor": "Adil"},
    {"day": "Wednesday", "time": "7:00 AM", "class": "Yoga", "instructor": "Taha "},
    {"day": "Wednesday", "time": "6:00 PM", "class": "Zumba", "instructor": "Anas"},
    {"day": "Thursday", "time": "8:00 AM", "class": "Spin", "instructor": "Immad"},
    {"day": "Thursday", "time": "7:00 PM", "class": "HIIT", "instructor": "Adil"},
    {"day": "Friday", "time": "7:00 AM", "class": "Yoga", "instructor": "Taha "},
    {"day": "Saturday", "time": "8:00 AM", "class": "Yoga", "instructor": "Taha "},
    {"day": "Saturday", "time": "9:00 AM", "class": "HIIT", "instructor": "Adil"},
    {"day": "Saturday", "time": "10:00 AM", "class": "Spin", "instructor": "Immad"},
]

MEMBERSHIP_PLANS = [
    {
        "tier": "Basic",
        "price": "$29/month",
        "includes": "Gym floor access, locker room access",
        "freeze_policy": "Can freeze once per year, up to 30 days",
    },
    {
        "tier": "Standard",
        "price": "$49/month",
        "includes": "Gym floor access, locker room access, unlimited group classes",
        "freeze_policy": "Can freeze twice per year, up to 30 days each",
    },
    {
        "tier": "Premium",
        "price": "$79/month",
        "includes": "Gym floor access, locker room access, unlimited group classes, 2 personal training sessions per month, guest passes",
        "freeze_policy": "Can freeze up to 3 times per year, up to 45 days each",
    },
]

TRAINERS = [
    {"name": "Taha ", "specialty": "Yoga and flexibility training", "availability": "Mornings, Mon/Wed/Fri/Sat"},
    {"name": "Adil", "specialty": "HIIT and strength conditioning", "availability": "Evenings and Saturday mornings"},
    {"name": "Immad", "specialty": "Spin and cardio endurance", "availability": "Tue/Thu mornings, Saturday"},
    {"name": "Anas", "specialty": "Zumba and dance fitness", "availability": "Wednesday evenings"},
]

POLICIES = {
    "class_cancellation": "Classes can be cancelled up to 2 hours before start time with no penalty. Late cancellations may count against monthly class limits for Standard members.",
    "membership_freeze": "Freeze requests must be submitted at least 3 days before the desired start date. See plan-specific freeze limits above.",
    "guest_policy": "Premium members get 2 guest passes per month. Basic and Standard members can purchase day passes for guests at $15 each.",
}