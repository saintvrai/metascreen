# app/utils.py

import streamlit as st
import re
from constants import DISPLAY_NAMES

_orig_selectbox = st.selectbox

def patch_selectbox():
    def selectbox(label, options, *args, **kwargs):
        """
        Перекрываем st.selectbox так, 
        чтобы отображать из DISPLAY_NAMES[option],
        но возвращать самим option (английский ключ).
        """
        # если пользователь не передал свой format_func, ставим наш
        kwargs.setdefault(
            'format_func',
            lambda opt: DISPLAY_NAMES.get(opt, opt)
        )
        return _orig_selectbox(label, options, *args, **kwargs)

    st.selectbox = selectbox

def parse_params_string(s: str):
    """Вытягивает [H K L P] из строки и возвращает словарь."""
    m = re.search(r'\[([^\]]+)\]', s)
    if not m: raise ValueError("…")
    vals = [float(x) for x in m.group(1).split()]
    if len(vals)!=4: raise ValueError("…")
    return dict(zip(['H','K','L','P'], vals))
