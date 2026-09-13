"""
Trustify AI - Decision Engine

Combines:
1. CIFAKE EfficientNet-B0 -> AI-generation signal
2. CASIA EfficientNet-B0  -> image manipulation signal

Final classes:
    REAL
    AI-GENERATED
    AI-EDITED / MANIPULATED
"""


# =========================================================
# THRESHOLDS
# =========================================================

# CIFAKE: strong AI-generation signal
CIFAKE_GENERATED_THRESHOLD = 0.80

# CASIA: strong manipulation signal
CASIA_EDITED_THRESHOLD = 0.70


# =========================================================
# DECISION FUNCTION
# =========================================================

def decide(
    cifake_fake_score,
    casia_edited_score
):
    """
    Decide final authenticity class.

    Parameters
    ----------
    cifake_fake_score : float
        CIFAKE FAKE probability between 0 and 1.

    casia_edited_score : float
        CASIA EDITED probability between 0 and 1.

    Returns
    -------
    dict
        Prediction, confidence, trust score,
        reason, explanation and evidence.
    """

    cifake_fake_percent = round(
        cifake_fake_score * 100,
        2
    )

    casia_edited_percent = round(
        casia_edited_score * 100,
        2
    )

    # =====================================================
    # 1. AI-GENERATED
    # =====================================================

    if cifake_fake_score >= CIFAKE_GENERATED_THRESHOLD:

        confidence = cifake_fake_score * 100

        trust_score = max(
            0,
            min(
                100,
                round(100 - confidence)
            )
        )

        reason = (
            "Strong AI-generation signal detected."
        )

        explanation = (
            f"The CIFAKE detector assigned a "
            f"{cifake_fake_percent}% probability to the "
            f"AI-generated class. This is above the "
            f"{CIFAKE_GENERATED_THRESHOLD * 100:.0f}% "
            f"detection threshold. "
            f"The Grad-CAM visualization highlights the "
            f"image regions that contributed most strongly "
            f"to this model prediction."
        )

        evidence = [
            "High AI-generation probability from CIFAKE.",
            "CIFAKE score exceeds the configured detection threshold.",
            "Grad-CAM identifies regions that influenced the prediction."
        ]

        return {
            "prediction": "AI-GENERATED",
            "confidence": round(confidence, 2),
            "trust_score": trust_score,
            "reason": reason,
            "explanation": explanation,
            "evidence": evidence,
            "cifake_score": cifake_fake_percent,
            "casia_score": casia_edited_percent
        }

    # =====================================================
    # 2. AI-EDITED / MANIPULATED
    # =====================================================

    if casia_edited_score >= CASIA_EDITED_THRESHOLD:

        confidence = casia_edited_score * 100

        trust_score = max(
            0,
            min(
                100,
                round(100 - confidence)
            )
        )

        reason = (
            "Strong image-manipulation signal detected."
        )

        explanation = (
            f"The CASIA detector assigned a "
            f"{casia_edited_percent}% probability to the "
            f"edited/manipulated class. This is above the "
            f"{CASIA_EDITED_THRESHOLD * 100:.0f}% "
            f"detection threshold. "
            f"The Grad-CAM visualization highlights the "
            f"regions that contributed most strongly "
            f"to the manipulation prediction."
        )

        evidence = [
            "High manipulation probability from CASIA.",
            "CASIA score exceeds the configured detection threshold.",
            "Grad-CAM identifies regions that influenced the prediction."
        ]

        return {
            "prediction": "AI-EDITED / MANIPULATED",
            "confidence": round(confidence, 2),
            "trust_score": trust_score,
            "reason": reason,
            "explanation": explanation,
            "evidence": evidence,
            "cifake_score": cifake_fake_percent,
            "casia_score": casia_edited_percent
        }

    # =====================================================
    # 3. REAL
    # =====================================================

    confidence = max(
        1 - cifake_fake_score,
        1 - casia_edited_score
    ) * 100

    trust_score = max(
        0,
        min(
            100,
            round(confidence)
        )
    )

    reason = (
        "No strong AI-generation or manipulation signal detected."
    )

    explanation = (
        f"The CIFAKE detector assigned a "
        f"{cifake_fake_percent}% AI-generation probability, "
        f"while the CASIA detector assigned a "
        f"{casia_edited_percent}% manipulation probability. "
        f"Neither score exceeded its configured detection "
        f"threshold. The image is therefore classified as "
        f"REAL by the current decision rules. "
        f"Grad-CAM can be used to visualize the regions "
        f"that contributed to the REAL classification."
    )

    evidence = [
        "CIFAKE score is below the AI-generation threshold.",
        "CASIA score is below the manipulation threshold.",
        "Grad-CAM identifies regions influencing the REAL classification."
    ]

    return {
        "prediction": "REAL",
        "confidence": round(confidence, 2),
        "trust_score": trust_score,
        "reason": reason,
        "explanation": explanation,
        "evidence": evidence,
        "cifake_score": cifake_fake_percent,
        "casia_score": casia_edited_percent
    }