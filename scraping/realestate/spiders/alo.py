import scrapy
import json
import os

config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'config', 'config.json')

with open(config_path) as f:
    config = json.load(f)

BASE_URL = config["base_url"]
REGION_ID = config["region_id"]
LOCATION_IDS = config["location_ids"]

class AloSpider(scrapy.Spider):
    name = "alo"
    allowed_domains = ["alo.bg"]
    start_urls = [
        f"{BASE_URL}?region_id={REGION_ID}&location_ids={LOCATION_IDS}"
    ]

    custom_settings = {
        "FEEDS": {
            "../../datasets/raw_data.csv": {
                "format": "csv",
                "encoding": "utf8",
                "overwrite": True,
            }
        }
    }

    def parse(self, response):
        # --- listtop-params ---
        for ad in response.css("div.listtop-params"):
            def get_field(listing, title):
                return listing.css(
                    f"div.ads-param-title:contains('{title}:') + div.ads-params-cell span.ads-params-single *::text"
                ).get(default="").strip()

            yield {
                "location": ad.css("div.listtop-item-address i::text").get(default="").strip(),
                "price": get_field(ad, "Цена"),
                "property_type": get_field(ad, "Вид на имота"),
                "size": get_field(ad, "Квадратура"),
                "construction_type": get_field(ad, "Вид строителство"),
                "year_built": get_field(ad, "Година на строителство"),
                "floor_number": get_field(ad, "Номер на етажа"),
                "floor_type": get_field(ad, "Етаж")
            }

        # --- listvip-params ---
        for ad in response.css("div.listvip-params"):
            def get_field(title):
                return ad.css(f"span.ads-params-multi[title='{title}']::text").get(default="").strip()
            
            def get_price_field(title):
                return ad.css(f"span.ads-params-multi[title='{title}'] span[style*='white-space: nowrap']::text"
                ).get(default="").strip()

            yield {
                "location": ad.css("div.listvip-item-address i::text").get(default="").strip(),
                "price": get_price_field("Цена"),
                "property_type": get_field("Вид на имота"),
                "size": get_field("Квадратура"),
                "construction_type": get_field("Вид строителство"),
                "year_built": get_field("Година на строителство"),
                "floor_number": get_field("Номер на етажа"),
                "floor_type": get_field("Етаж")
            }

        # --- Pagination ---
        next_page_url = response.css("a[rel='next']::attr(href)").get()
        if next_page_url:
            yield response.follow(next_page_url, callback=self.parse)
