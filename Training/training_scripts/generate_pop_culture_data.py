"""
generate_pop_culture_data.py - build-your-own-model kit demo on world pop culture
(music, film, sports, business, media) plus companies, platforms, leagues/awards,
and world places (cities and countries).

!!! STARTER DATA - VERIFY EVERY FACT AGAINST A CURRENT SOURCE BEFORE TRAINING !!!
This domain is open and fast-changing and any author's knowledge has a cutoff, so
this file uses only ROCK-STABLE public facts: what someone is famous for as a
profession, their field, (musicians) primary genre / (athletes) primary sport,
the country they are from, and for places their country/continent/capital. NO
ages, finances, relationships, current teams, current projects, or "latest"
anything. Primary genre/sport only (some span more than one).
"""
import sys
from collections import Counter
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "Training Material + Pre-trained Models",
                                "Training Materials", "build_your_own_model"))
from corpus_lib import list_join, run

TOPIC = "world pop culture"

# field, role (profession), from (country), known_for (broad/stable), and:
#   genre -> musicians   sport -> athletes
ENTITIES = [
    # ---- Music ----
    {"name": "Taylor Swift",      "field": "music", "role": "singer-songwriter", "genre": "pop",       "from": "the United States", "known_for": "pop and country songwriting and record-breaking concert tours"},
    {"name": "Beyonce",           "field": "music", "role": "singer",            "genre": "R&B",       "from": "the United States", "known_for": "R&B and pop music and elaborate live performances"},
    {"name": "Drake",             "field": "music", "role": "rapper",            "genre": "hip-hop",   "from": "Canada",            "known_for": "chart-topping hip-hop singles"},
    {"name": "Kendrick Lamar",    "field": "music", "role": "rapper",            "genre": "hip-hop",   "from": "the United States", "known_for": "critically acclaimed hip-hop"},
    {"name": "Billie Eilish",     "field": "music", "role": "singer",            "genre": "pop",       "from": "the United States", "known_for": "alternative pop music"},
    {"name": "Rihanna",           "field": "music", "role": "singer",            "genre": "pop",       "from": "Barbados",          "known_for": "pop and R&B hits and a beauty and fashion business"},
    {"name": "Bruno Mars",        "field": "music", "role": "singer",            "genre": "pop",       "from": "the United States", "known_for": "pop and funk-influenced hit songs"},
    {"name": "Adele",             "field": "music", "role": "singer",            "genre": "pop",       "from": "England",           "known_for": "powerful pop ballads"},
    {"name": "Ariana Grande",     "field": "music", "role": "singer",            "genre": "pop",       "from": "the United States", "known_for": "pop music and a wide vocal range"},
    {"name": "The Weeknd",        "field": "music", "role": "singer",            "genre": "R&B",       "from": "Canada",            "known_for": "R&B and pop music"},
    {"name": "Jay-Z",             "field": "music", "role": "rapper",            "genre": "hip-hop",   "from": "the United States", "known_for": "hip-hop and building a business empire"},
    {"name": "Snoop Dogg",        "field": "music", "role": "rapper",            "genre": "hip-hop",   "from": "the United States", "known_for": "West Coast hip-hop"},
    {"name": "Dolly Parton",      "field": "music", "role": "singer-songwriter", "genre": "country",   "from": "the United States", "known_for": "country music and songwriting"},
    {"name": "Post Malone",       "field": "music", "role": "singer",            "genre": "hip-hop",   "from": "the United States", "known_for": "genre-blending hip-hop and pop"},
    {"name": "SZA",               "field": "music", "role": "singer",            "genre": "R&B",       "from": "the United States", "known_for": "R&B and soul music"},
    {"name": "Olivia Rodrigo",    "field": "music", "role": "singer-songwriter", "genre": "pop",       "from": "the United States", "known_for": "pop and pop-rock songwriting"},
    {"name": "Ed Sheeran",        "field": "music", "role": "singer-songwriter", "genre": "pop",       "from": "England",           "known_for": "acoustic pop songwriting"},
    {"name": "Dua Lipa",          "field": "music", "role": "singer",            "genre": "pop",       "from": "England",           "known_for": "dance-pop music"},
    {"name": "Bad Bunny",         "field": "music", "role": "singer",            "genre": "reggaeton", "from": "Puerto Rico",       "known_for": "Latin trap and reggaeton"},
    {"name": "Shakira",           "field": "music", "role": "singer",            "genre": "pop",       "from": "Colombia",          "known_for": "Latin pop music and dancing"},
    {"name": "Eminem",            "field": "music", "role": "rapper",            "genre": "hip-hop",   "from": "the United States", "known_for": "fast, technical rap"},
    {"name": "Justin Bieber",     "field": "music", "role": "singer",            "genre": "pop",       "from": "Canada",            "known_for": "pop music since he was a teenager"},
    {"name": "Lady Gaga",         "field": "music", "role": "singer",            "genre": "pop",       "from": "the United States", "known_for": "theatrical pop music and acting"},
    {"name": "Elton John",        "field": "music", "role": "singer-songwriter", "genre": "rock",      "from": "England",           "known_for": "piano-driven rock and pop over many decades"},
    {"name": "Bruce Springsteen", "field": "music", "role": "singer-songwriter", "genre": "rock",      "from": "the United States", "known_for": "heartland rock and long live shows"},
    {"name": "Stevie Wonder",     "field": "music", "role": "singer-songwriter", "genre": "R&B",       "from": "the United States", "known_for": "soul and R&B classics"},
    {"name": "Usher",             "field": "music", "role": "singer",            "genre": "R&B",       "from": "the United States", "known_for": "R&B music and dancing"},
    {"name": "Coldplay",          "field": "music", "role": "rock band",         "genre": "rock",      "from": "England",           "known_for": "arena rock and pop anthems"},
    # ---- Film & TV ----
    {"name": "Tom Hanks",         "field": "film", "role": "actor",              "from": "the United States", "known_for": "leading roles in many acclaimed films"},
    {"name": "Meryl Streep",      "field": "film", "role": "actor",              "from": "the United States", "known_for": "a wide range of acclaimed film roles"},
    {"name": "Denzel Washington", "field": "film", "role": "actor",              "from": "the United States", "known_for": "powerful dramatic film roles"},
    {"name": "Zendaya",           "field": "film", "role": "actor",              "from": "the United States", "known_for": "film and television roles"},
    {"name": "Leonardo DiCaprio", "field": "film", "role": "actor",              "from": "the United States", "known_for": "leading roles in major films"},
    {"name": "Morgan Freeman",    "field": "film", "role": "actor",              "from": "the United States", "known_for": "distinguished film roles and a famous voice"},
    {"name": "Scarlett Johansson","field": "film", "role": "actor",              "from": "the United States", "known_for": "leading film roles"},
    {"name": "Robert Downey Jr.", "field": "film", "role": "actor",              "from": "the United States", "known_for": "leading roles in major films"},
    {"name": "Dwayne Johnson",    "field": "film", "role": "actor",              "from": "the United States", "known_for": "action films after a wrestling career"},
    {"name": "Jennifer Lawrence", "field": "film", "role": "actor",              "from": "the United States", "known_for": "leading film roles"},
    {"name": "Will Smith",        "field": "film", "role": "actor",              "from": "the United States", "known_for": "film roles and an early music career"},
    {"name": "Viola Davis",       "field": "film", "role": "actor",              "from": "the United States", "known_for": "acclaimed film and stage roles"},
    {"name": "Kevin Hart",        "field": "film", "role": "comedian and actor", "from": "the United States", "known_for": "stand-up comedy and comedy films"},
    {"name": "Brad Pitt",         "field": "film", "role": "actor",              "from": "the United States", "known_for": "leading film roles and producing"},
    {"name": "Margot Robbie",     "field": "film", "role": "actor",              "from": "Australia",         "known_for": "leading film roles and producing"},
    {"name": "Ryan Reynolds",     "field": "film", "role": "actor",              "from": "Canada",            "known_for": "comedy and action film roles"},
    {"name": "Cate Blanchett",    "field": "film", "role": "actor",              "from": "Australia",         "known_for": "a wide range of acclaimed film roles"},
    {"name": "Samuel L. Jackson", "field": "film", "role": "actor",              "from": "the United States", "known_for": "roles in a huge number of films"},
    {"name": "Steven Spielberg",  "field": "film", "role": "film director",      "from": "the United States", "known_for": "directing many landmark blockbuster films"},
    {"name": "Christopher Nolan", "field": "film", "role": "film director",      "from": "England",           "known_for": "directing ambitious, mind-bending films"},
    {"name": "Martin Scorsese",   "field": "film", "role": "film director",      "from": "the United States", "known_for": "directing acclaimed crime and drama films"},
    # ---- Sports ----
    {"name": "LeBron James",      "field": "sports", "role": "basketball player", "sport": "basketball", "from": "the United States", "known_for": "being one of the greatest NBA players"},
    {"name": "Stephen Curry",     "field": "sports", "role": "basketball player", "sport": "basketball", "from": "the United States", "known_for": "revolutionizing three-point shooting"},
    {"name": "Michael Jordan",    "field": "sports", "role": "basketball player", "sport": "basketball", "from": "the United States", "known_for": "being widely called the greatest basketball player ever"},
    {"name": "Shaquille O'Neal",  "field": "sports", "role": "basketball player", "sport": "basketball", "from": "the United States", "known_for": "a dominant NBA center career"},
    {"name": "Kevin Durant",      "field": "sports", "role": "basketball player", "sport": "basketball", "from": "the United States", "known_for": "being one of the best scorers in the NBA"},
    {"name": "Serena Williams",   "field": "sports", "role": "tennis player",     "sport": "tennis",     "from": "the United States", "known_for": "dominating professional tennis"},
    {"name": "Novak Djokovic",    "field": "sports", "role": "tennis player",     "sport": "tennis",     "from": "Serbia",            "known_for": "winning a record number of major tennis titles"},
    {"name": "Rafael Nadal",      "field": "sports", "role": "tennis player",     "sport": "tennis",     "from": "Spain",             "known_for": "his dominance on clay courts"},
    {"name": "Roger Federer",     "field": "sports", "role": "tennis player",     "sport": "tennis",     "from": "Switzerland",       "known_for": "an elegant, record-setting tennis career"},
    {"name": "Tom Brady",         "field": "sports", "role": "football player",   "sport": "football",   "from": "the United States", "known_for": "a record-setting NFL quarterback career"},
    {"name": "Patrick Mahomes",   "field": "sports", "role": "football player",   "sport": "football",   "from": "the United States", "known_for": "being a star NFL quarterback"},
    {"name": "Lionel Messi",      "field": "sports", "role": "soccer player",     "sport": "soccer",     "from": "Argentina",         "known_for": "being one of the greatest soccer players ever"},
    {"name": "Cristiano Ronaldo", "field": "sports", "role": "soccer player",     "sport": "soccer",     "from": "Portugal",          "known_for": "a record-breaking soccer goalscoring career"},
    {"name": "Kylian Mbappe",     "field": "sports", "role": "soccer player",     "sport": "soccer",     "from": "France",            "known_for": "his speed and goalscoring in soccer"},
    {"name": "Simone Biles",      "field": "sports", "role": "gymnast",           "sport": "gymnastics", "from": "the United States", "known_for": "being one of the greatest gymnasts"},
    {"name": "Tiger Woods",       "field": "sports", "role": "golfer",            "sport": "golf",       "from": "the United States", "known_for": "one of the greatest golf careers"},
    {"name": "Usain Bolt",        "field": "sports", "role": "sprinter",          "sport": "track",      "from": "Jamaica",           "known_for": "being the fastest sprinter in history"},
    {"name": "Michael Phelps",    "field": "sports", "role": "swimmer",           "sport": "swimming",   "from": "the United States", "known_for": "winning the most Olympic gold medals ever"},
    {"name": "Lewis Hamilton",    "field": "sports", "role": "racing driver",     "sport": "racing",     "from": "England",           "known_for": "a record-tying number of Formula 1 titles"},
    {"name": "Shohei Ohtani",     "field": "sports", "role": "baseball player",   "sport": "baseball",   "from": "Japan",             "known_for": "excelling as both a pitcher and a hitter"},
    # ---- Business & tech ----
    {"name": "Elon Musk",         "field": "business", "role": "entrepreneur",         "from": "South Africa",      "known_for": "leading Tesla and SpaceX"},
    {"name": "Jeff Bezos",        "field": "business", "role": "entrepreneur",         "from": "the United States", "known_for": "founding Amazon"},
    {"name": "Mark Zuckerberg",   "field": "business", "role": "technology executive", "from": "the United States", "known_for": "co-founding and leading Meta, the company behind Facebook"},
    {"name": "Bill Gates",        "field": "business", "role": "entrepreneur",         "from": "the United States", "known_for": "co-founding Microsoft and later philanthropy"},
    {"name": "Tim Cook",          "field": "business", "role": "technology executive", "from": "the United States", "known_for": "leading Apple as its chief executive"},
    {"name": "Warren Buffett",    "field": "business", "role": "investor",             "from": "the United States", "known_for": "being one of the most successful investors"},
    {"name": "Sam Altman",        "field": "business", "role": "technology executive", "from": "the United States", "known_for": "leading the artificial intelligence company OpenAI"},
    {"name": "Jensen Huang",      "field": "business", "role": "technology executive", "from": "Taiwan",            "known_for": "leading the chip company Nvidia"},
    # ---- Media ----
    {"name": "Oprah Winfrey",     "field": "media", "role": "television host",              "from": "the United States", "known_for": "a long-running talk show and a media empire"},
    {"name": "Jimmy Fallon",      "field": "media", "role": "television host",              "from": "the United States", "known_for": "hosting a late-night talk show"},
    {"name": "MrBeast",           "field": "media", "role": "YouTuber and content creator", "from": "the United States", "known_for": "elaborate YouTube stunts and giveaways"},
]

# Cities and countries - rock-stable geography (country / continent / capital).
PLACES = [
    {"name": "New York",       "kind": "city", "country": "the United States", "continent": "North America", "known_for": "being the largest city in the United States and a global center for finance and culture"},
    {"name": "Los Angeles",    "kind": "city", "country": "the United States", "continent": "North America", "known_for": "the film and entertainment industry"},
    {"name": "London",         "kind": "city", "country": "England",           "continent": "Europe",        "known_for": "being the capital of England and the United Kingdom"},
    {"name": "Paris",          "kind": "city", "country": "France",            "continent": "Europe",        "known_for": "being the capital of France and famous for art and fashion"},
    {"name": "Tokyo",          "kind": "city", "country": "Japan",             "continent": "Asia",          "known_for": "being the capital of Japan and one of the largest cities in the world"},
    {"name": "Rome",           "kind": "city", "country": "Italy",             "continent": "Europe",        "known_for": "being the capital of Italy and its ancient history"},
    {"name": "Sydney",         "kind": "city", "country": "Australia",         "continent": "Oceania",       "known_for": "its harbour and opera house"},
    {"name": "Toronto",        "kind": "city", "country": "Canada",            "continent": "North America", "known_for": "being the largest city in Canada"},
    {"name": "Rio de Janeiro", "kind": "city", "country": "Brazil",            "continent": "South America", "known_for": "its beaches and Carnival festival"},
    {"name": "Dubai",          "kind": "city", "country": "the United Arab Emirates", "continent": "Asia",  "known_for": "its skyscrapers and luxury shopping"},
    {"name": "Cairo",          "kind": "city", "country": "Egypt",             "continent": "Africa",        "known_for": "being the capital of Egypt and near the ancient pyramids"},
    {"name": "Mumbai",         "kind": "city", "country": "India",             "continent": "Asia",          "known_for": "being India's largest city and its film industry"},
    {"name": "United States",  "kind": "country", "capital": "Washington, D.C.", "continent": "North America", "known_for": "being a large country known for its economy and culture"},
    {"name": "Canada",         "kind": "country", "capital": "Ottawa",           "continent": "North America", "known_for": "its size and natural landscapes"},
    {"name": "Mexico",         "kind": "country", "capital": "Mexico City",      "continent": "North America", "known_for": "its history, food, and culture"},
    {"name": "England",        "kind": "country", "capital": "London",           "continent": "Europe",        "known_for": "its history and being part of the United Kingdom"},
    {"name": "France",         "kind": "country", "capital": "Paris",            "continent": "Europe",        "known_for": "its art, food, and history"},
    {"name": "Italy",          "kind": "country", "capital": "Rome",             "continent": "Europe",        "known_for": "its art, food, and ancient history"},
    {"name": "Germany",        "kind": "country", "capital": "Berlin",           "continent": "Europe",        "known_for": "its engineering and economy"},
    {"name": "Spain",          "kind": "country", "capital": "Madrid",           "continent": "Europe",        "known_for": "its culture, food, and football"},
    {"name": "Japan",          "kind": "country", "capital": "Tokyo",            "continent": "Asia",          "known_for": "its technology and culture"},
    {"name": "China",          "kind": "country", "capital": "Beijing",          "continent": "Asia",          "known_for": "being the most populous country and a large economy"},
    {"name": "India",          "kind": "country", "capital": "New Delhi",        "continent": "Asia",          "known_for": "its large population and diverse culture"},
    {"name": "Brazil",         "kind": "country", "capital": "Brasilia",         "continent": "South America", "known_for": "being the largest country in South America and its love of football"},
    {"name": "Australia",      "kind": "country", "capital": "Canberra",         "continent": "Oceania",       "known_for": "being both a country and a continent"},
    {"name": "Egypt",          "kind": "country", "capital": "Cairo",            "continent": "Africa",        "known_for": "the ancient pyramids and the Nile river"},
]

ATTRIBUTE_QUESTIONS = {
    "role": {"ask": ["What does {name} do?", "What is {name}'s profession?",
                     "What is {name}'s job?", "What is {name} known as?"],
             "answer": "{name} is a {value}."},
    "field": {"ask": ["What field is {name} in?", "What industry is {name} in?",
                      "What area is {name} famous in?"],
              "answer": "{name} works in {value}."},
    "from": {"ask": ["Where is {name} from?", "What country is {name} from?",
                     "Where does {name} come from?"],
             "answer": "{name} is from {value}."},
    "known_for": {"ask": ["What is {name} known for?", "Why is {name} famous?",
                          "What made {name} famous?", "What is {name} famous for?"],
                  "answer": "{name} is known for {value}."},
    "genre": {"ask": ["What genre is {name}?", "What kind of music does {name} make?",
                      "What style of music is {name}?"],
              "answer": "{name} is primarily a {value} artist."},
    "sport": {"ask": ["What sport does {name} play?", "What does {name} play?",
                      "What sport is {name} known for?"],
              "answer": "{name} plays {value}."},
}

REVERSE_LOOKUPS = {
    "field": {"ask": ["Which famous people are in {value}?", "List people in {value}.",
                      "Name some {value} celebrities."],
              "answer": "In {value}: {list}."},
    "role": {"ask": ["Which of these people are {value}s?", "List the {value}s.",
                     "Name the {value}s."],
             "answer": "The {value}s here are {list}."},
    "genre": {"ask": ["Which musicians make {value} music?", "List the {value} artists.",
                      "Name some {value} musicians."],
              "answer": "The {value} artists here are {list}."},
    "sport": {"ask": ["Which athletes play {value}?", "List the {value} players.",
                      "Name the {value} athletes."],
              "answer": "In {value}: {list}."},
    "from": {"ask": ["Which famous people are from {value}?", "Who here is from {value}?",
                     "Name some celebrities from {value}."],
             "answer": "From {value}: {list}."},
}

SPORT_INFO = {
    "basketball": "Basketball is a team sport where two teams of five players try to score by shooting a ball through a raised hoop.",
    "tennis":     "Tennis is a racket sport where players hit a ball back and forth over a net, scoring when an opponent can't return it.",
    "football":   "American football is a team sport where players advance an oval ball by running and passing it to score touchdowns.",
    "soccer":     "Soccer, called football in most countries, is a team sport where players use their feet to move a ball into the other team's goal.",
    "gymnastics": "Gymnastics is a sport of strength, balance, and agility performed on events like the floor, balance beam, and bars.",
    "golf":       "Golf is a sport where players use clubs to hit a ball into a series of holes in as few strokes as possible.",
    "track":      "Track and field is a set of running, jumping, and throwing events, including sprinting races won by the fastest runner.",
    "swimming":   "Swimming is a sport where athletes race through water using strokes like freestyle, backstroke, and butterfly.",
    "racing":     "Motor racing, such as Formula 1, is a sport where drivers race high-speed cars around a track to finish first.",
    "baseball":   "Baseball is a bat-and-ball team sport where a batter tries to hit a pitched ball and run around four bases to score.",
}
GENRE_INFO = {
    "pop":       "Pop is catchy, mainstream music made for a wide audience, usually with strong melodies and simple, repeatable structures.",
    "hip-hop":   "Hip-hop, also called rap, is built on rhythmic spoken vocals over beats, often with sampled or electronic production.",
    "R&B":       "R&B, short for rhythm and blues, features soulful vocals and smooth grooves, rooted in soul and gospel music.",
    "country":   "Country is rooted in American folk and Western music, often with guitars, storytelling lyrics, and everyday-life themes.",
    "rock":      "Rock is guitar-driven music with a strong beat, ranging from soft ballads to loud, energetic anthems.",
    "reggaeton": "Reggaeton is a Latin genre built on a steady dance beat, blending reggae, hip-hop, and Latin styles, usually sung in Spanish.",
}

PLATFORMS = [
    {"name": "TikTok",    "kind": "social media platform",   "purpose": "sharing short vertical videos",
     "desc": "TikTok is a social media app for short vertical videos, known for viral trends, dances, and music clips."},
    {"name": "Instagram", "kind": "social media platform",   "purpose": "sharing photos and short videos",
     "desc": "Instagram is a social media app for sharing photos and short videos."},
    {"name": "YouTube",   "kind": "video platform",          "purpose": "watching and uploading videos",
     "desc": "YouTube is a video platform where people watch and upload everything from music videos to tutorials and vlogs."},
    {"name": "X",         "kind": "social media platform",   "purpose": "posting short public messages",
     "desc": "X, formerly Twitter, is a social media platform for short public text posts, news, and conversation."},
    {"name": "Facebook",  "kind": "social media platform",   "purpose": "connecting with friends and family",
     "desc": "Facebook is a social network for connecting with friends and family and joining groups, owned by Meta."},
    {"name": "Snapchat",  "kind": "social media platform",   "purpose": "sharing photos and videos that disappear",
     "desc": "Snapchat is a social media app for sharing photos and videos that disappear after they are viewed."},
    {"name": "Reddit",    "kind": "social media platform",   "purpose": "discussing topics in community forums",
     "desc": "Reddit is a social media site organized into community forums where people post, vote, and discuss almost any topic."},
    {"name": "Twitch",    "kind": "video platform",          "purpose": "watching live streams",
     "desc": "Twitch is a video platform for watching people live-stream, especially video games and creative work."},
    {"name": "Spotify",   "kind": "music streaming service", "purpose": "streaming music and podcasts",
     "desc": "Spotify is a music streaming service for listening to songs and podcasts on demand."},
    {"name": "Netflix",   "kind": "video streaming service", "purpose": "streaming shows and movies",
     "desc": "Netflix is a streaming service for watching TV shows and movies on demand over the internet."},
]

COMPANIES = [
    {"name": "Tesla",     "kind": "car and energy company",         "from": "the United States", "desc": "Tesla is an American company that makes electric cars and clean energy products."},
    {"name": "SpaceX",    "kind": "space company",                  "from": "the United States", "desc": "SpaceX is an American company that builds rockets and spacecraft for space travel."},
    {"name": "Amazon",    "kind": "technology and retail company",  "from": "the United States", "desc": "Amazon is an American company known for online shopping and cloud computing."},
    {"name": "Meta",      "kind": "technology company",             "from": "the United States", "desc": "Meta is the American technology company that owns Facebook and Instagram."},
    {"name": "Microsoft", "kind": "technology company",             "from": "the United States", "desc": "Microsoft is an American technology company known for its computer software."},
    {"name": "Apple",     "kind": "technology company",             "from": "the United States", "desc": "Apple is an American technology company known for its phones and computers."},
    {"name": "Google",    "kind": "technology company",             "from": "the United States", "desc": "Google is an American technology company known for its search engine and online services."},
    {"name": "Nvidia",    "kind": "technology company",             "from": "the United States", "desc": "Nvidia is an American technology company known for making computer chips for graphics and artificial intelligence."},
    {"name": "Disney",    "kind": "entertainment company",          "from": "the United States", "desc": "Disney is an American entertainment company known for movies, theme parks, and cartoon characters."},
    {"name": "Nike",      "kind": "sportswear company",             "from": "the United States", "desc": "Nike is an American company known for athletic shoes and sportswear."},
    {"name": "Sony",      "kind": "technology and entertainment company", "from": "Japan",        "desc": "Sony is a Japanese company known for electronics, video games, and movies."},
    {"name": "Nintendo",  "kind": "video game company",             "from": "Japan",             "desc": "Nintendo is a Japanese company known for video games and game consoles."},
    {"name": "Samsung",   "kind": "technology company",             "from": "South Korea",       "desc": "Samsung is a South Korean company known for phones, televisions, and electronics."},
]

# Leagues and awards - stable cultural institutions. (name, questions, answer)
INSTITUTIONS = [
    ("NBA", ["What is the NBA?", "What does NBA stand for?", "Tell me about the NBA."],
     "The NBA, or National Basketball Association, is the top professional basketball league in North America."),
    ("NFL", ["What is the NFL?", "What does NFL stand for?", "Tell me about the NFL."],
     "The NFL, or National Football League, is the top professional American football league in the United States."),
    ("MLB", ["What is MLB?", "What does MLB stand for?", "Tell me about Major League Baseball."],
     "MLB, or Major League Baseball, is the top professional baseball league in North America."),
    ("Premier League", ["What is the Premier League?", "Tell me about the Premier League."],
     "The Premier League is the top professional soccer league in England and one of the most popular in the world."),
    ("Grammys", ["What are the Grammys?", "What is a Grammy?", "Tell me about the Grammy Awards."],
     "The Grammy Awards are the biggest awards in the music industry, given each year for outstanding recordings."),
    ("Oscars", ["What are the Oscars?", "What is an Oscar?", "Tell me about the Academy Awards."],
     "The Academy Awards, or Oscars, are the most prestigious awards in film, given each year for outstanding movies."),
    ("Emmys", ["What are the Emmys?", "Tell me about the Emmy Awards."],
     "The Emmy Awards are the top awards for television, given each year for outstanding shows and performances."),
    ("Olympics", ["What are the Olympics?", "Tell me about the Olympic Games."],
     "The Olympics are a major international sports competition held every few years, where athletes from around the world compete."),
    ("World Cup", ["What is the World Cup?", "Tell me about the soccer World Cup."],
     "The World Cup is the biggest international soccer tournament, held every four years between national teams."),
]

FIELD_INFO = {
    "music":    (["What is the music industry?", "Tell me about the music business."],
                 "The music industry is the business of creating, recording, and selling music and live performances."),
    "film":     (["What is the film industry?", "Tell me about Hollywood."],
                 "The film and television industry produces movies and TV shows for theaters and streaming services."),
    "sports":   (["What are professional sports?", "Tell me about pro sports."],
                 "Professional sports are organized athletic competitions where athletes compete in leagues and tournaments."),
    "business": (["What is the tech industry?", "Tell me about the business world."],
                 "The business and technology world is the companies and leaders who build products and services, from cars to social media."),
    "media":    (["What is the media industry?", "Tell me about the media business."],
                 "The media industry includes television, streaming, talk shows, and online creators who make and host content."),
}

CAPABILITIES = [
    (["What can you do?", "What do you know?", "What is this?", "Help", "What can I ask you?"],
     "I can answer questions about world pop culture: famous musicians, actors, athletes, and business figures, "
     "plus music genres, sports, companies, social media, streaming, and world cities and countries."),
]

LORE = [
    (["Tell me about social media.", "What is social media?", "How do celebrities reach fans today?"],
     "Social media platforms like Instagram, TikTok, YouTube, and X let people share photos, videos, and short "
     "posts. They shape how celebrities reach fans and how trends spread."),
    (["Tell me about streaming.", "What is streaming?", "How do people watch and listen today?"],
     "Streaming services deliver music and video over the internet on demand. Platforms like Netflix, Spotify, "
     "and YouTube changed how people watch shows and listen to music."),
    (["What is an influencer?", "Tell me about content creators.", "What is a social media influencer?"],
     "A social media influencer, or content creator, builds an audience by posting online, often on YouTube, "
     "TikTok, or Instagram, and can shape trends and opinions much like a traditional celebrity."),
    (["What is a continent?", "Name the continents.", "What are the continents?"],
     "A continent is one of the world's large land areas. The seven continents are Africa, Antarctica, Asia, "
     "Europe, North America, Oceania, and South America."),
]


def build(c):
    for e in ENTITIES:
        name = e["name"]
        role_desc = f"{e['genre']} {e['role']}" if "genre" in e else e["role"]
        c.qa_variants([f"Who is {name}?", f"Tell me about {name}.",
                       f"What can you tell me about {name}?"],
                      f"{name} is a {role_desc} known for {e['known_for']}.")
        for attr, spec in ATTRIBUTE_QUESTIONS.items():
            if attr not in e:
                continue
            c.qa_variants([q.format(name=name) for q in spec["ask"]],
                          spec["answer"].format(name=name, value=e[attr]))

    for attr, spec in REVERSE_LOOKUPS.items():
        buckets = {}
        for e in ENTITIES:
            if attr in e:
                buckets.setdefault(e[attr], []).append(e["name"])
        for value, names in buckets.items():
            c.qa_variants([q.format(value=value) for q in spec["ask"]],
                          spec["answer"].format(value=value, list=list_join(names)))

    for sport in sorted({e["sport"] for e in ENTITIES if "sport" in e}):
        info = SPORT_INFO.get(sport)
        if info:
            c.qa_variants([f"What is {sport}?", f"Describe {sport}.",
                           f"How does {sport} work?", f"What kind of sport is {sport}?"], info)
    for genre in sorted({e["genre"] for e in ENTITIES if "genre" in e}):
        info = GENRE_INFO.get(genre)
        if info:
            c.qa_variants([f"What is {genre}?", f"What is {genre} music?",
                           f"Describe {genre} music.", f"What is {genre} music like?"], info)

    # Places: cities and countries with country / continent / capital facts.
    for p in PLACES:
        nm = p["name"]
        c.qa_variants([f"What is {nm}?", f"Tell me about {nm}.", f"Where is {nm}?"],
                      f"{nm} is a {p['kind']} in {p['continent']} known for {p['known_for']}.")
        c.qa_variants([f"What continent is {nm} in?", f"Where in the world is {nm}?"],
                      f"{nm} is in {p['continent']}.")
        if p["kind"] == "city":
            c.qa_variants([f"What country is {nm} in?", f"Where is the city of {nm}?"],
                          f"{nm} is a city in {p['country']}.")
        if p["kind"] == "country" and "capital" in p:
            c.qa_variants([f"What is the capital of {nm}?", f"What is {nm}'s capital city?"],
                          f"The capital of {nm} is {p['capital']}.")
    for kind in ("city", "country"):
        plural = "cities" if kind == "city" else "countries"
        by_cont = {}
        for p in PLACES:
            if p["kind"] == kind:
                by_cont.setdefault(p["continent"], []).append(p["name"])
        for cont, nms in by_cont.items():
            c.qa_variants([f"Which {plural} are in {cont}?", f"List {plural} in {cont}.",
                           f"Name some {plural} in {cont}."],
                          f"In {cont}: {list_join(nms)}.")

    for p in PLATFORMS:
        nm = p["name"]
        c.qa_variants([f"What is {nm}?", f"Describe {nm}.", f"Tell me about {nm}."], p["desc"])
        c.qa_variants([f"What is {nm} for?", f"What do people use {nm} for?", f"What do you do on {nm}?"],
                      f"{nm} is for {p['purpose']}.")
    kinds = {}
    for p in PLATFORMS:
        kinds.setdefault(p["kind"], []).append(p["name"])
    for kind, nms in kinds.items():
        c.qa_variants([f"Which are {kind}s?", f"List the {kind}s.", f"Name some {kind}s."],
                      f"The {kind}s here are {list_join(nms)}.")

    for co in COMPANIES:
        nm = co["name"]
        c.qa_variants([f"What is {nm}?", f"Describe {nm}.", f"Tell me about {nm}."], co["desc"])
        c.qa_variants([f"What kind of company is {nm}?", f"What does {nm} do?"],
                      f"{nm} is a {co['kind']}.")
        if "from" in co:
            c.qa_variants([f"What country is {nm} from?", f"Where is {nm} based?"],
                          f"{nm} is a company from {co['from']}.")

    for _nm, questions, answer in INSTITUTIONS:
        c.qa_variants(questions, answer)

    for _field, (questions, answer) in FIELD_INFO.items():
        c.qa_variants(questions, answer)

    for questions, answer in CAPABILITIES:
        c.qa_variants(questions, answer)

    for questions, passage in LORE:
        c.prose(passage)
        c.qa_variants(questions, passage)

    # Whole-word tokens: EVERY proper noun in an answer must be one token or a tiny
    # BPE tokenizer shreds it. People + places + platforms + companies + the
    # leagues/awards (short + long forms) + punctuated genres + place bits.
    inst_tokens = ["NBA", "National Basketball Association", "NFL", "National Football League",
                   "MLB", "Major League Baseball", "Premier League", "Grammys", "Grammy Awards",
                   "Oscars", "Academy Awards", "Emmys", "Emmy Awards", "Olympics", "World Cup"]
    extra_tokens = ["R&B", "hip-hop", "West Coast", "United Kingdom", "OpenAI", "Formula 1",
                    "Washington, D.C.", "Mexico City", "New Delhi", "the United Arab Emirates",
                    "Twitter", "YouTuber", "Olympic", "Antarctica", "Japanese", "Korean", "Spanish"]
    place_bits = []
    for p in PLACES:
        place_bits += [p.get("country", ""), p["continent"], p.get("capital", "")]
    from_vals = [e.get("from", "") for e in ENTITIES] + [co.get("from", "") for co in COMPANIES]
    return (
        [e["name"] for e in ENTITIES]
        + [p["name"] for p in PLACES]
        + [p["name"] for p in PLATFORMS]
        + [co["name"] for co in COMPANIES]
        + inst_tokens + extra_tokens
        + [b for b in place_bits if b]
        + [v for v in from_vals if v]
    )


if __name__ == "__main__":
    run(build)
