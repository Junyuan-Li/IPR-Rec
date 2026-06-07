class LocalFeedbackEngine:
    def build_prompt(self, history_titles, candidate_title: str) -> str:
        history_text = '; '.join(history_titles[-5:]) if history_titles else 'N/A'
        return (
            'User reading history:\n'
            f'{history_text}\n\n'
            'Candidate news:\n'
            f'{candidate_title}\n\n'
            'Answer only YES or NO.'
        )

    def generate(self, prompt: str, max_tokens: int = 10) -> str:
        del prompt, max_tokens
        raise RuntimeError('Local LLM backend is not configured. Use ONEREC_LLM_BACKEND=dashscope or provide a local engine implementation.')

    def plan_path(self, prompt: str, max_tokens: int = 50) -> str:
        return self.generate(prompt, max_tokens=max_tokens)


def get_feedback_engine():
    return LocalFeedbackEngine()