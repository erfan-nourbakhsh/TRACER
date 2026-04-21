
import random
from typing import Dict, List


RECOVERY_TEMPLATES = {
    "oscillation": [
        "I notice some changes in your preferences. Just to confirm, "
        "you'd like a {domain} with {slot_text}?",
        "Let me verify: for the {domain}, you want {slot_text}, correct?",
        "I want to make sure I have this right. Your {domain} preference "
        "is {slot_text}. Is that correct?",
    ],
    "coverage_gap": [
        "To complete your {domain} request, I still need the following: "
        "{missing_slots}. Could you provide those?",
        "I have {filled_text} for the {domain}. I also need {missing_slots} "
        "to proceed. Could you help with that?",
        "For the {domain}, I'm still missing {missing_slots}. "
        "Would you like to provide those details?",
    ],
    "conflict": [
        "I want to make sure I have this right. For the {domain}, "
        "your preference is {slot_text}. Is that correct?",
        "Just to clarify, you'd like {slot_text} for the {domain}?",
    ],
    "domain_confusion": [
        "Just to clarify, are we still looking for a {domain}, "
        "or would you like to focus on something else?",
        "I want to make sure I'm helping you with the right thing. "
        "Are you looking for a {domain}?",
    ],
    "general": [
        "Let me confirm what I have so far: {slot_text} for the {domain}. "
        "Is everything correct?",
        "Just to make sure I have everything right for the {domain}: "
        "{slot_text}. Shall I proceed?",
    ],
}


def _determine_failure_type(features):
    osc, cov, conf, vel, shifts = features

    if osc >= 2:
        return "oscillation"
    if cov < 0.4:
        return "coverage_gap"
    if conf >= 2:
        return "conflict"
    if shifts >= 3:
        return "domain_confusion"
    return "general"


def _format_slot_text(belief_state, domain=None):
    parts = []
    for slot, value in sorted(belief_state.items()):
        slot_name = slot.split("-", 1)[-1] if "-" in slot else slot
        slot_name = slot_name.replace("_", " ").replace("book ", "")
        parts.append(f"{slot_name}: {value}")
    return ", ".join(parts) if parts else "no preferences specified yet"


def _get_missing_slots(belief_state, required_slots, domain):
    missing = []
    for slot in required_slots:
        full_key = f"{domain}-{slot}"
        if full_key not in belief_state or not belief_state[full_key]:
            slot_name = slot.replace("_", " ").replace("book ", "")
            missing.append(slot_name)
    return missing


def generate_template_recovery(features, belief_state, domain,
                                required_slots=None):
    failure_type = _determine_failure_type(features)

    slot_text = _format_slot_text(belief_state, domain)
    missing_slots = _get_missing_slots(
        belief_state, required_slots or [], domain
    )
    missing_text = ", ".join(missing_slots) if missing_slots else "nothing"

    filled_slots = {k: v for k, v in belief_state.items()
                    if k.startswith(f"{domain}-") and v}
    filled_text = _format_slot_text(filled_slots, domain)

    templates = RECOVERY_TEMPLATES.get(failure_type, RECOVERY_TEMPLATES["general"])
    template = random.choice(templates)

    return template.format(
        domain=domain,
        slot_text=slot_text,
        missing_slots=missing_text,
        filled_text=filled_text,
    )
