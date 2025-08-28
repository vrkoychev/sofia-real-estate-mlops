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

    def parse(self, response):
        # --- listtop-params ---
        listtop_listings = response.css("div.listtop-params")
        for ad in listtop_listings:
            yield {
                "type": "listtop",
                "title": ad.css("a::text").get(default="").strip(),
                "neighbourhood": ad.css("div.listtop-item-address i::text").get(default="").strip(),
                "price": ad.css("div.ads-param-title:contains('Цена:') + div.ads-params-cell span.ads-params-single span::text").get(default="").strip(),
                "price_per_sqm": ad.css("div.ads-param-title:contains('за кв.м:') + div.ads-params-cell span.ads-params-single span::text").get(default="").strip(),
                "property_type": ad.css("div.ads-param-title:contains('Вид на имота:') + div.ads-params-cell span.ads-params-single::text").get(default="").strip(),
                "size": ad.css("div.ads-param-title:contains('Квадратура:') + div.ads-params-cell span.ads-params-single::text").get(default="").strip(),
                "construction_type": ad.css("div.ads-param-title:contains('Вид строителство:') + div.ads-params-cell span.ads-params-single::text").get(default="").strip(),
                "year_built": ad.css("div.ads-param-title:contains('Година на строителство:') + div.ads-params-cell span.ads-params-single::text").get(default="").strip(),
                "floor_number": ad.css("div.ads-param-title:contains('Номер на етажа:') + div.ads-params-cell span.ads-params-single::text").get(default="").strip(),
                "floor_type": ad.css("div.ads-param-title:contains('Етаж:') + div.ads-params-cell span.ads-params-single::text").get(default="").strip(),
                "url": response.url
            }

        # --- listvip-params ---
        listvip_listings = response.css("div.listvip-params")
        for ad in listvip_listings:
            # Map fields by title attribute
            def get_field(title):
                return ad.css(f"span.ads-params-multi[title='{title}']::text").get(default="").strip()

            yield {
                "type": "listvip",
                "title": ad.css("h3.listvip-item-title::text").get(default="").strip(),
                "neighbourhood": ad.css("div.listvip-item-address i::text").get(default="").strip(),
                "price": get_field("Цена"),
                "price_per_sqm": get_field("за кв.м"),
                "property_type": get_field("Вид на имота"),
                "size": get_field("Квадратура"),
                "construction_type": get_field("Вид строителство"),
                "year_built": get_field("Година на строителство"),
                "floor_number": get_field("Номер на етажа"),
                "floor_type": get_field("Етаж"),
                "url": response.css("div.listvip-item-header a::attr(href)").get(default="").strip()
            }

        # --- Pagination ---
        next_page = response.css("a.next::attr(href)").get()
        if next_page:
            yield response.follow(next_page, callback=self.parse)
