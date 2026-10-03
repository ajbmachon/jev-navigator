"""Secret masking scans repeated request text once without retaining other requests."""

from collections import Counter

from jev_navigator.judgments.secrets import SecretMasker, mask_by_content


def test_repeated_request_text_is_scanned_once_without_retaining_other_requests(monkeypatch):
    masker = SecretMasker()
    mask = masker.mask
    discover = masker.masked_values
    masked = Counter()
    discovered = Counter()

    def observe_mask(self, text):
        masked[text] += 1
        return mask(text)

    def observe_discovery(self, text):
        discovered[text] += 1
        return discover(text)

    monkeypatch.setattr(SecretMasker, "mask", observe_mask)
    monkeypatch.setattr(SecretMasker, "masked_values", observe_discovery)
    reference = 'send("order-hook-4f7a1c")'
    value = {
        "assignment": 'WEBHOOK_TOKEN = "order-hook-4f7a1c"',
        "rows": [{"code": reference} for _ in range(200)],
    }

    result = mask_by_content(value, masker)

    assert result == {
        "assignment": 'WEBHOOK_TOKEN = "[MASKED]"',
        "rows": [{"code": 'send("[MASKED]")'} for _ in range(200)],
    }
    assert discovered[reference] == 1
    assert masked[reference] == 1
    # Without the assignment this ordinary string is not secret-shaped. The previous request's
    # discovered values must not survive as hidden state in a later masking operation.
    assert mask_by_content({"code": reference}, masker) == {"code": reference}
