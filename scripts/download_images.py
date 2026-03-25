from icrawler.builtin import BingImageCrawler

queries = [
"dog running",
"cat sleeping",
"monkey eating banana",
"horse running",
"lion in jungle",

"person riding bike",
"person driving car",
"man playing guitar",
"woman cooking",
"people walking street",

"car at night",
"sports car on highway",
"motorcycle racing",
"bus on road",
"train moving",

"airplane flying",
"helicopter in sky",
"rocket launch",
"ship in ocean",
"boat on lake",

"mountain landscape",
"beach sunset",
"waterfall nature",
"forest road",
"snowy mountain",

"city skyline",
"busy street market",
"night city lights",
"bridge over river",
"skyscrapers downtown",

"football match",
"basketball game",
"cricket stadium",
"tennis player hitting ball",
"people swimming",

"pizza on table",
"burger and fries",
"fruit basket",
"coffee cup on desk",
"cake dessert",

"laptop on desk",
"keyboard and mouse",
"person using smartphone",
"coding on computer",
"office workspace",

"dog playing with ball",
"cat sitting on sofa",
"bird flying in sky",
"cow grazing field",
"elephant walking"
]

for q in queries:

    folder = "images/" + q.replace(" ", "_")

    crawler = BingImageCrawler(storage={'root_dir': folder})

    print("Downloading:", q)

    crawler.crawl(
        keyword=q,
        max_num=10
    )

print("Download complete")