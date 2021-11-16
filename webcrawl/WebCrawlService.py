import math
import requests
from bs4 import BeautifulSoup

def get_movie_title(movie_code):
    url = 'https://movie.naver.com/movie/bi/mi/basic.naver?code={}'.format(movie_code)
    result = requests.get(url)
    doc = BeautifulSoup(result.text, 'html.parser')

    title = doc.select('h3.h_movie > a')[0].get_text()
    return title



def calc_pages(movie_code):
    url = 'https://movie.naver.com/movie/bi/mi/basic.naver?code={}'.format(movie_code)
    result = requests.get(url)
    doc = BeautifulSoup(result.text, 'html.parser')
    all_count = doc.select('strong.total > em')[0].get_text().strip()
    pages = math.ceil(int(all_count) / 10)
    return pages

def get_reviews(movie_code,pages, title):
    count = 0  # total review count

    for page in range(1, pages+1):
        new_url = 'https://movie.naver.com/movie/bi/mi/pointWriteFormList.naver?code={}&type=after&isActualPointWriteExecute=false&isMileageSubscriptionAlready=false&isMileageSubscriptionReject=false&page={}'.format(movie_code,page)
        result = requests.get(new_url)
        doc = BeautifulSoup(result.text, 'html.parser')

        review_list = doc.select('div.score_result > ul > li')

        for one in review_list:
            count += 1
            print('## USER -> {} #######################################################'.format(count))

            # 평점정보 수집
            score = one.select('div.star_score > em')[0].get_text()
            # 리뷰정보 수집
            review = one.select('div.score_reple > p > span')[-1].get_text().strip()

            # 작성자(닉네임) 정보 수집
            original_writer = one.select('div.score_reple dt em')[0].get_text().strip()
            idx_end = original_writer.find('(')  # index를 알려주는게 find 함수이다.
            writer = original_writer[0:idx_end]

            # 날짜 정보 수집
            original_date = one.select('div.score_reple dt em')[1].get_text()
            # yyyy.MM.dd 전처리 코드 작성
            # time_no = original_date.find(' ')
            # date = original_date[0:time_no]
            date = original_date[:10]

            # yyyy.mm.dd 전처리 코드 작성
            print('TITLE:{}'.format(title))
            print('REVIEW:{}'.format(review))
            print('WRITER:{}'.format(writer))
            print('SCORE:{}'.format(score))
            print('DATE:{}'.format(date))