from abc import ABC, abstractmethod


class BaseSummariser:

    @abstractmethod
    def generate_summary(self):
        pass
