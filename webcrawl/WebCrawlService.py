import requests
from bs4 import BeautifulSoup

def get_movie_title(movie_code):
    url = 'https://movie.naver.com/movie/bi/mi/basic.naver?code={}'.format(movie_code)
    result = requests.get(url)
    doc = BeautifulSoup(result.text, 'html.parser')

    title = doc.select('h3.h_movie > a')[0].get_text()
    return title