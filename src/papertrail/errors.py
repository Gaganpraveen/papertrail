class PaperTrailError(Exception):
    """An actionable failure safe to show at the command line."""


class SourceError(PaperTrailError):
    pass


class ParseError(PaperTrailError):
    pass


class ModelError(PaperTrailError):
    pass


class GroundingError(PaperTrailError):
    pass
