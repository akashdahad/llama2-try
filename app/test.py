from bs4 import BeautifulSoup
import requests

# https://www.listofcompaniesin.com/index-p75834.html
companies = []
for i in range(75833):
  html = requests.get(f"https://www.listofcompaniesin.com/index-p{i+1}html").text
  print("URL:", f"https://www.listofcompaniesin.com/index-p{i+1}.html" )
  soup = BeautifulSoup(html, 'html.parser')
  print(soup)
  res = soup.find_all(["li"])
  print("Company Details Retrieved:",  len(companies))
  for li_item in res:
    company = {}
    if li_item.h4 == None:
      continue
    else:
        company['title'] = li_item.h4.a.string 
        company['link'] = li_item.h4.a['href']
        company['description'] = li_item.find('p', class_='txt').string
        if(len(li_item.find_all('span')) == 2):
          company['telephone'] = str(li_item.find_all('span')[0]).replace('<em>Telephone：</em>', '').replace('<span>', '').replace('</span>', '').strip()
          company['address'] = str(li_item.find_all('span')[1]).replace('<em>Address</em>', '').replace('<span>', '').replace('</span>', '').strip()
        else:
          if len(li_item.find_all('span')) == 1:
            company['telephone'] = str(li_item.find_all('span')[0]).replace('<em>Telephone：</em>', '').replace('<span>', '').replace('</span>', '').strip()
        companies.append(company)
        if(li_item.find('span') == None):
          continue

for company in companies:
  print('------------------------------')
  print(company)
    