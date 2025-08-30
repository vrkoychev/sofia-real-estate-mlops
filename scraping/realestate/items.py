# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy

class RealEstateItem(scrapy.Item):
    location = scrapy.Field()
    price = scrapy.Field()
    property_type = scrapy.Field()
    size = scrapy.Field()
    construction_type = scrapy.Field()
    year_built = scrapy.Field()
    floor_number = scrapy.Field()
    floor_type = scrapy.Field()

