import csv
book_html='/Users/khushsi/Downloads/book.html'
try:
    from bs4 import BeautifulSoup as BS
except ImportError:
    from BeautifulSoup import BeautifulSoup

hfile = open(book_html)
soup = BS(hfile)
divi = soup.find('div',attrs={"id":"page-container"})
# elem = soup.body.findAll('div',attrs={'class':"pf w3 h13"})
elem = soup.body.findAll('div',attrs={'class':"pf w0 h0"})
fwrite = csv.writer(open('final_book.csv','w'))

for row in elem:

    if(row.text.strip() != "" and len(row.text.strip()) > 50):
        print(1)
        fwrite.writerow(["book-"+row['data-page-no'],row.text])


# for i in range(1,3):
#     divip = divi.find('div',attrs={'class':"pf w3 h13",'data-page-no':'1'})
#     print(i)
#     print(divip.text)


