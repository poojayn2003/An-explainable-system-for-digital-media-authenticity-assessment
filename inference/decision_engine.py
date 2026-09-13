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

# ---------------------------------------------------------
# THRESHOLDS
# ---------------------------------------------------------

# CIFAKE: strong AI-generation signal
CIFAKE_GENERATED_THRESHOLD = 0.80

# CASIA: strong manipulation signal
CASIA_EDITED_THRESHOLD = 0.70


def decide(cifake_fake_score, casia_edited_score):
    """
    Decide the final authenticity class.

    Parameters
    ----------
    cifake_fake_score : float
        CIFAKE FAKE score between 0 and 1.

    casia_edited_score : float
        CASIA EDITED score between 0 and 1.

    Returns
    -------
    dict
        Final prediction, confidence, trust score and evidence.
    """

    # -----------------------------------------------------
    # 1. Strong AI-generation signal
    # -----------------------------------------------------
    if cifake_fake_score >= CIFAKE_GENERATED_THRESHOLD:

        confidence = cifake_fake_score * 100
        trust_score = max(0, min(100, round(100 - confidence)))

        return {
            "prediction": "AI-GENERATED",
            "confidence": round(confidence, 2),
            "trust_score": trust_score,
            "reason": "Strong AI-generation signal detected.",
            "cifake_score": round(cifake_fake_score * 100, 2),
            "casia_score": round(casia_edited_score * 100, 2)
        }

    # -----------------------------------------------------
    # 2. Strong manipulation signal
    # -----------------------------------------------------
    if casia_edited_score >= CASIA_EDITED_THRESHOLD:

        confidence = casia_edited_score * 100
        trust_score = max(0, min(100, round(100 - confidence)))

        return {
            "prediction": "AI-EDITED / MANIPULATED",
            "confidence": round(confidence, 2),
            "trust_score": trust_score,
            "reason": "Strong image-manipulation signal detected.",
            "cifake_score": round(cifake_fake_score * 100, 2),
            "casia_score": round(casia_edited_score * 100, 2)
        }

    # -----------------------------------------------------
    # 3. No strong detection signal
    # -----------------------------------------------------
    confidence = max(
        1 - cifake_fake_score,
        1 - casia_edited_score
    ) * 100

    trust_score = max(0, min(100, round(confidence)))

    return {
        "prediction": "REAL",
        "confidence": round(confidence, 2),
        "trust_score": trust_score,
        "reason": "No strong AI-generation or manipulation signal detected.",
        "cifake_score": round(cifake_fake_score * 100, 2),
        "casia_score": round(casia_edited_score * 100, 2)
    }