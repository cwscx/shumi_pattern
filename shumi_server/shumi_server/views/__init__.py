from .main_views import index_view
from .api_views import save_baby_data, delete_baby_action
from .ai_views import get_prediction, get_gemini_insights

__all__ = [
    'index_view',
    'save_baby_data',
    'delete_baby_action',
    'get_prediction',
    'get_gemini_insights'
]