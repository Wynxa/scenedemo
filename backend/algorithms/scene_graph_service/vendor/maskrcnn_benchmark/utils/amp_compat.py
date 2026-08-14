class _AmpFallback:
    @staticmethod
    def float_function(func):
        return func


try:
    from apex import amp as apex_amp
    amp = apex_amp
except Exception:
    amp = _AmpFallback()
