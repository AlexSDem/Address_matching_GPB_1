"""Run: python -m streamlit run app.py"""
from io import BytesIO
from dataclasses import asdict
import pandas as pd
import streamlit as st
from src.hybrid import HybridAddressMatcher
from src.io import ROOT,read_reference,read_aliases

st.set_page_config(page_title='Сопоставление адресов',page_icon='📍',layout='wide')
st.title('Сопоставление адресов')
st.caption('Введите адрес с опечатками или сокращениями — получите совпадение либо запрос на уточнение.')

@st.cache_resource(show_spinner='Подготовка справочника…')
def load_matcher(data,threshold,margin):
    ref=read_reference(BytesIO(data))
    return HybridAddressMatcher(threshold=threshold,min_margin=margin).fit(ref,read_aliases())

with st.sidebar:
    st.header('Справочник')
    upload=st.file_uploader('CSV со справочником',type=['csv'])
    st.caption('Колонка address или united_addr обязательна. Рекомендуются id, region, city, street_type, street, house, building, structure.')
    with st.expander('Настройки принятия'):
        threshold=st.slider('Минимальный score',0.,1.,.84,.01)
        margin=st.slider('Отрыв от второго кандидата',0.,.3,.06,.01)
        st.caption('Score — мера сходства, не вероятность. Пороги требуют проверки на ваших данных.')
try:
    payload=upload.getvalue() if upload is not None else (ROOT/'data/reference_demo.csv').read_bytes()
    matcher=load_matcher(payload,threshold,margin)
except (ValueError,UnicodeDecodeError,pd.errors.ParserError) as exc:
    st.error(f'Не удалось прочитать справочник: {exc}')
    st.stop()
if upload is None:
    st.info('Демонстрационный справочник: 17 синтетических записей. Загрузите свой CSV для работы с вашими адресами.')
st.caption(f'Записей в справочнике: {len(matcher.ref):,}')

single,batch=st.tabs(['Один адрес','Пакетная обработка'])
with single:
    with st.form('address_form'):
        query=st.text_input('Адрес',value='Красноярск, ул. Ленина 25')
        submitted=st.form_submit_button('Найти адрес',type='primary')
    if submitted:
        if not query.strip():
            st.warning('Введите адрес.')
        else:
            result=matcher.match_one(query)
            if result.status=='accepted':
                st.success(result.best)
            elif result.status=='review':
                st.warning('Нужно уточнение. Проверьте регион, номер дома, корпус и написание улицы.')
                st.write('Кандидат для проверки:',result.candidate)
            else:
                st.error('Надёжное совпадение не найдено. Проверьте адрес или полноту справочника.')
            st.metric('Сходство',f'{result.score:.3f}')
            with st.expander('Разбор и причины решения'):
                st.json(result.parsed)
                st.write('Причины:',', '.join(result.reasons) or 'Проверки пройдены')
                st.caption('Технические названия причин описаны в README.')
            with st.expander('Кандидаты для диагностики'):
                _,rows=matcher.rank(query)
                st.dataframe(pd.DataFrame(rows[:5]).drop(columns=['components']),hide_index=True)
with batch:
    queries=st.file_uploader('CSV с колонкой query',type=['csv'],key='queries')
    if queries is not None and st.button('Обработать CSV'):
        try:
            frame=read_reference(queries)
            if 'query' not in frame: raise ValueError('Нужна колонка query')
            if len(frame)>10000: raise ValueError('Для интерактивной обработки максимум 10 000 строк')
            result=matcher.match_batch(frame['query'])
            st.dataframe(result.drop(columns=['parsed']),hide_index=True)
            st.download_button('Скачать результаты',result.to_csv(index=False).encode('utf-8-sig'),'matches.csv','text/csv')
        except (ValueError,UnicodeDecodeError,pd.errors.ParserError) as exc:
            st.error(str(exc))
