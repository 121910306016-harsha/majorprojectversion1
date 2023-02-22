# import requests
# import json

# def suggest_hospitals(latitude, longitude, radius):
#     api_key = "AIzaSyDLtlMd6N1SararhW3zXcFrdnDEHbKyoGM"
#     endpoint = "https://maps.googleapis.com/maps/api/place/nearbysearch/json?"
#     location = f"{latitude},{longitude}"
#     query = "hospital"
#     request = f"{endpoint}location={location}&radius={radius}&type={query}&key={api_key}"
#     print(request)
#     response = requests.get(request)
#     response_json = json.loads(response.text)

#     hospitals = []
#     for place in response_json["results"]:
#         name = place["name"]
#         latitude = place["geometry"]["location"]["lat"]
#         longitude = place["geometry"]["location"]["lng"]
#         hospitals.append((name, latitude, longitude))
# latitude = 37.7749
# longitude = -122.4194
# radius = 5000
# hospitals = suggest_hospitals(latitude, longitude, radius)
# print("Nearest Hospitals:")
# for hospital in hospitals:
#     print(f"- {hospital[0]} ({hospital[1]}, {hospital[2]})")
import requests
from bs4 import BeautifulSoup
def suggest_hospitals(latitude, longitude, radius):
    endpoint = "https://www.google.com/maps/dir/"
    location = f"{latitude},{longitude}"
    query = "/search/hospital"
    url = f"{endpoint}{location}{query}"
    print(url)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    print(soup)
    hospitals = []
    l=soup.find_all("div", class_="section-result-content-container")
    print(l)
    for place in soup.find_all("div", class_="section-result-content-container"):
        print(place)
        name = place.find("h3", class_="section-result-title").text
        address = place.find("span", class_="section-result-location").text
        hospitals.append((name, address))

    return hospitals

if __name__ == "__main__":
    latitude = 17.782550
    longitude = 83.376600
    radius = 5000
    hospitals = suggest_hospitals(latitude, longitude, radius)
    print("Nearest Hospitals:")
    for hospital in hospitals:
        print(f"- {hospital[0]} ({hospital[1]})")