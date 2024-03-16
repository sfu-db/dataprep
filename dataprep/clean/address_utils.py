"""
Constants used by the clean_address() and validate_address() functions
"""

# pylint: disable=C0301, C0302, E1101

from builtins import zip
from builtins import str
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict
import string
import re
import os
import warnings
import pycrfsuite

TAG_MAPPING = {
    "OccupancyType": "apartment",
    "OccupancyIdentifier": "apartment",
    "SubaddressType": "apartment",
    "SubaddressIdentifier": "apartment",
    "BuildingName": "building",
    "AddressNumber": "house_number",
    "StreetNamePreDirectional": "street_prefix",
    "StreetName": "street_name",
    "StreetNamePostType": "street_suffix",
    "PlaceName": "city",
    "StateName": "state",
    "ZipCode": "zipcode",
}

KEYWORDS = [
    "house_number",
    "street_prefix_abbr",
    "street_prefix_full",
    "street_name",
    "street_suffix_abbr",
    "street_suffix_full",
    "city",
    "state_abbr",
    "state_full",
    "zipcode",
    "building",
    "apartment",
]

PREFIXES = {
    "n": "N.",
    "e": "E.",
    "s": "S.",
    "w": "W.",
    "ne": "NE.",
    "nw": "NW.",
    "se": "SE.",
    "sw": "SW.",
    "north": "N.",
    "east": "E.",
    "south": "S.",
    "west": "W.",
    "northeast": "NE.",
    "northwest": "NW.",
    "southeast": "SE.",
    "southwest": "SW.",
}

FULL_PREFIX = {
    "N.": "North",
    "E.": "East",
    "S.": "South",
    "W.": "West",
    "NE.": "North East",
    "NW.": "North West",
    "SE.": "South East",
    "SW.": "South West",
}

FULL_STATES = {
    "Mississippi": "MS",
    "Oklahoma": "OK",
    "Delaware": "DE",
    "Minnesota": "MN",
    "Illinois": "IL",
    "Arkansas": "AR",
    "New Mexico": "NM",
    "Indiana": "IN",
    "Maryland": "MD",
    "Louisiana": "LA",
    "Idaho": "ID",
    "Wyoming": "WY",
    "Tennessee": "TN",
    "Arizona": "AZ",
    "Iowa": "IA",
    "Michigan": "MI",
    "Kansas": "KS",
    "Utah": "UT",
    "Virginia": "VA",
    "Oregon": "OR",
    "Connecticut": "CT",
    "Montana": "MT",
    "California": "CA",
    "Massachusetts": "MA",
    "West Virginia": "WV",
    "South Carolina": "SC",
    "New Hampshire": "NH",
    "Wisconsin": "WI",
    "Vermont": "VT",
    "Georgia": "GA",
    "North Dakota": "ND",
    "Pennsylvania": "PA",
    "Florida": "FL",
    "Alaska": "AK",
    "Kentucky": "KY",
    "Hawaii": "HI",
    "Nebraska": "NE",
    "Missouri": "MO",
    "Ohio": "OH",
    "Alabama": "AL",
    "New York": "NY",
    "South Dakota": "SD",
    "Colorado": "CO",
    "New Jersey": "NJ",
    "Washington": "WA",
    "North Carolina": "NC",
    "District of Columbia": "DC",
    "Texas": "TX",
    "Nevada": "NV",
    "Maine": "ME",
    "Rhode Island": "RI",
}

ABBR_STATES = {state_abbr: state_full for state_full, state_abbr in FULL_STATES.items()}

SUFFIXES = {
    "ALLEE": ("ALY", "ALLEY"),
    "ALLEY": ("ALY", "ALLEY"),
    "ALLY": ("ALY", "ALLEY"),
    "ALY": ("ALY", "ALLEY"),
    "ANEX": ("ANX", "ANEX"),
    "ANNEX": ("ANX", "ANEX"),
    "ANNX": ("ANX", "ANEX"),
    "ANX": ("ANX", "ANEX"),
    "ARC": ("ARC", "ARCADE"),
    "ARCADE": ("ARC", "ARCADE"),
    "AV": ("AVE", "AVENUE"),
    "AVE": ("AVE", "AVENUE"),
    "AVEN": ("AVE", "AVENUE"),
    "AVENU": ("AVE", "AVENUE"),
    "AVENUE": ("AVE", "AVENUE"),
    "AVN": ("AVE", "AVENUE"),
    "AVNUE": ("AVE", "AVENUE"),
    "BAYOO": ("BYU", "BAYOU"),
    "BAYOU": ("BYU", "BAYOU"),
    "BCH": ("BCH", "BEACH"),
    "BEACH": ("BCH", "BEACH"),
    "BEND": ("BND", "BEND"),
    "BND": ("BND", "BEND"),
    "BLF": ("BLF", "BLUFF"),
    "BLUF": ("BLF", "BLUFF"),
    "BLUFF": ("BLF", "BLUFF"),
    "BLUFFS": ("BLFS", "BLUFFS"),
    "BOT": ("BTM", "BOTTOM"),
    "BOTTM": ("BTM", "BOTTOM"),
    "BOTTOM": ("BTM", "BOTTOM"),
    "BTM": ("BTM", "BOTTOM"),
    "BLVD": ("BLVD", "BOULEVARD"),
    "BOUL": ("BLVD", "BOULEVARD"),
    "BOULEVARD": ("BLVD", "BOULEVARD"),
    "BOULV": ("BLVD", "BOULEVARD"),
    "BR": ("BR", "BRANCH"),
    "BRANCH": ("BR", "BRANCH"),
    "BRNCH": ("BR", "BRANCH"),
    "BRDGE": ("BRG", "BRIDGE"),
    "BRG": ("BRG", "BRIDGE"),
    "BRIDGE": ("BRG", "BRIDGE"),
    "BRK": ("BRK", "BROOK"),
    "BROOK": ("BRK", "BROOK"),
    "BROOKS": ("BRKS", "BROOKS"),
    "BURG": ("BG", "BURG"),
    "BURGS": ("BGS", "BURGS"),
    "BYP": ("BYP", "BYPASS"),
    "BYPA": ("BYP", "BYPASS"),
    "BYPAS": ("BYP", "BYPASS"),
    "BYPASS": ("BYP", "BYPASS"),
    "BYPS": ("BYP", "BYPASS"),
    "CAMP": ("CP", "CAMP"),
    "CMP": ("CP", "CAMP"),
    "CP": ("CP", "CAMP"),
    "CANYN": ("CYN", "CANYON"),
    "CANYON": ("CYN", "CANYON"),
    "CNYN": ("CYN", "CANYON"),
    "CYN": ("CYN", "CANYON"),
    "CAPE": ("CPE", "CAPE"),
    "CPE": ("CPE", "CAPE"),
    "CAUSEWAY": ("CSWY", "CAUSEWAY"),
    "CAUSWAY": ("CSWY", "CAUSEWAY"),
    "CSWY": ("CSWY", "CAUSEWAY"),
    "CEN": ("CTR", "CENTER"),
    "CENT": ("CTR", "CENTER"),
    "CENTER": ("CTR", "CENTER"),
    "CENTR": ("CTR", "CENTER"),
    "CENTRE": ("CTR", "CENTER"),
    "CNTER": ("CTR", "CENTER"),
    "CNTR": ("CTR", "CENTER"),
    "CTR": ("CTR", "CENTER"),
    "CENTERS": ("CTRS", "CENTERS"),
    "CIR": ("CIR", "CIRCLE"),
    "CIRC": ("CIR", "CIRCLE"),
    "CIRCL": ("CIR", "CIRCLE"),
    "CIRCLE": ("CIR", "CIRCLE"),
    "CRCL": ("CIR", "CIRCLE"),
    "CRCLE": ("CIR", "CIRCLE"),
    "CIRCLES": ("CIR", "CIRCLES"),
    "CLF": ("CLF", "CLIFF"),
    "CLIFF": ("CLF", "CLIFF"),
    "CLFS": ("CLFS", "CLIFFS"),
    "CLIFFS": ("CLFS", "CLIFFS"),
    "CLB": ("CLB", "CLUB"),
    "CLUB": ("CLB", "CLUB"),
    "COMMON": ("CMN", "COMMON"),
    "COR": ("COR", "CORNER"),
    "CORNER": ("COR", "CORNER"),
    "CORNERS": ("CORS", "CORNERS"),
    "CORS": ("CORS", "CORNERS"),
    "COURSE": ("CRSE", "COURSE"),
    "CRSE": ("CRSE", "COURSE"),
    "COURT": ("CT", "COURT"),
    "CRT": ("CT", "COURT"),
    "CT": ("CT", "COURT"),
    "COURTS": ("CTS", "COURTS"),
    "COVE": ("CV", "COVE"),
    "CV": ("CV", "COVE"),
    "COVES": ("CVS", "COVES"),
    "CK": ("CRK", "CREEK"),
    "CR": ("CRK", "CREEK"),
    "CREEK": ("CRK", "CREEK"),
    "CRK": ("CRK", "CREEK"),
    "CRECENT": ("CRES", "CRESCENT"),
    "CRES": ("CRES", "CRESCENT"),
    "CRESCENT": ("CRES", "CRESCENT"),
    "CRESENT": ("CRES", "CRESCENT"),
    "CRSCNT": ("CRES", "CRESCENT"),
    "CRSENT": ("CRES", "CRESCENT"),
    "CRSNT": ("CRES", "CRESCENT"),
    "CREST": ("CREST", "CREST"),
    "CROSSING": ("XING", "CROSSING"),
    "CRSSING": ("XING", "CROSSING"),
    "CRSSNG": ("XING", "CROSSING"),
    "XING": ("XING", "CROSSING"),
    "CROSSROAD": ("XRD", "CROSSROAD"),
    "CURVE": ("CURV", "CURVE"),
    "DALE": ("DL", "DALE"),
    "DL": ("DL", "DALE"),
    "DAM": ("DM", "DAM"),
    "DM": ("DM", "DAM"),
    "DIV": ("DV", "DIVIDE"),
    "DIVIDE": ("DV", "DIVIDE"),
    "DV": ("DV", "DIVIDE"),
    "DVD": ("DV", "DIVIDE"),
    "DR": ("DR", "DRIVE"),
    "DRIV": ("DR", "DRIVE"),
    "DRIVE": ("DR", "DRIVE"),
    "DRV": ("DR", "DRIVE"),
    "DRIVES": ("DRS", "DRIVES"),
    "EST": ("EST", "ESTATE"),
    "ESTATE": ("EST", "ESTATE"),
    "ESTATES": ("ESTS", "ESTATES"),
    "ESTS": ("ESTS", "ESTATES"),
    "EXP": ("EXPY", "EXPRESSWAY"),
    "EXPR": ("EXPY", "EXPRESSWAY"),
    "EXPRESS": ("EXPY", "EXPRESSWAY"),
    "EXPRESSWAY": ("EXPY", "EXPRESSWAY"),
    "EXPW": ("EXPY", "EXPRESSWAY"),
    "EXPY": ("EXPY", "EXPRESSWAY"),
    "EXT": ("EXT", "EXTENSION"),
    "EXTENSION": ("EXT", "EXTENSION"),
    "EXTN": ("EXT", "EXTENSION"),
    "EXTNSN": ("EXT", "EXTENSION"),
    "EXTENSIONS": ("EXTS", "EXTENSIONS"),
    "EXTS": ("EXTS", "EXTENSIONS"),
    "FALL": ("FALL", "FALL"),
    "FALLS": ("FLS", "FALLS"),
    "FLS": ("FLS", "FALLS"),
    "FERRY": ("FRY", "FERRY"),
    "FRRY": ("FRY", "FERRY"),
    "FRY": ("FRY", "FERRY"),
    "FIELD": ("FLD", "FIELD"),
    "FLD": ("FLD", "FIELD"),
    "FIELDS": ("FLDS", "FIELDS"),
    "FLDS": ("FLDS", "FIELDS"),
    "FLAT": ("FLT", "FLAT"),
    "FLT": ("FLT", "FLAT"),
    "FLATS": ("FLTS", "FLATS"),
    "FLTS": ("FLTS", "FLATS"),
    "FORD": ("FRD", "FORD"),
    "FRD": ("FRD", "FORD"),
    "FORDS": ("FRDS", "FORDS"),
    "FOREST": ("FRST", "FOREST"),
    "FORESTS": ("FRST", "FOREST"),
    "FRST": ("FRST", "FOREST"),
    "FORG": ("FRG", "FORGE"),
    "FORGE": ("FRG", "FORGE"),
    "FRG": ("FRG", "FORGE"),
    "FORGES": ("FRGS", "FORGES"),
    "FORK": ("FRK", "FORK"),
    "FRK": ("FRK", "FORK"),
    "FORKS": ("FRKS", "FORKS"),
    "FRKS": ("FRKS", "FORKS"),
    "FORT": ("FT", "FORT"),
    "FRT": ("FT", "FORT"),
    "FT": ("FT", "FORT"),
    "FREEWAY": ("FWY", "FREEWAY"),
    "FREEWY": ("FWY", "FREEWAY"),
    "FRWAY": ("FWY", "FREEWAY"),
    "FRWY": ("FWY", "FREEWAY"),
    "FWY": ("FWY", "FREEWAY"),
    "GARDEN": ("GDN", "GARDEN"),
    "GARDN": ("GDN", "GARDEN"),
    "GDN": ("GDN", "GARDEN"),
    "GRDEN": ("GDN", "GARDEN"),
    "GRDN": ("GDN", "GARDEN"),
    "GARDENS": ("GDNS", "GARDENS"),
    "GDNS": ("GDNS", "GARDENS"),
    "GRDNS": ("GDNS", "GARDENS"),
    "GATEWAY": ("GTWY", "GATEWAY"),
    "GATEWY": ("GTWY", "GATEWAY"),
    "GATWAY": ("GTWY", "GATEWAY"),
    "GTWAY": ("GTWY", "GATEWAY"),
    "GTWY": ("GTWY", "GATEWAY"),
    "GLEN": ("GLN", "GLEN"),
    "GLN": ("GLN", "GLEN"),
    "GLENS": ("GLNS", "GLENS"),
    "GREEN": ("GRN", "GREEN"),
    "GRN": ("GRN", "GREEN"),
    "GREENS": ("GRNS", "GREENS"),
    "GROV": ("GRV", "GROVE"),
    "GROVE": ("GRV", "GROVE"),
    "GRV": ("GRV", "GROVE"),
    "GROVES": ("GRVS", "GROVES"),
    "HARB": ("HBR", "HARBOR"),
    "HARBOR": ("HBR", "HARBOR"),
    "HARBR": ("HBR", "HARBOR"),
    "HBR": ("HBR", "HARBOR"),
    "HRBOR": ("HBR", "HARBOR"),
    "HARBORS": ("HBRS", "HARBORS"),
    "HAVEN": ("HVN", "HAVEN"),
    "HAVN": ("HVN", "HAVEN"),
    "HVN": ("HVN", "HAVEN"),
    "HEIGHT": ("HTS", "HEIGHTS"),
    "HEIGHTS": ("HTS", "HEIGHTS"),
    "HGTS": ("HTS", "HEIGHTS"),
    "HT": ("HTS", "HEIGHTS"),
    "HTS": ("HTS", "HEIGHTS"),
    "HIGHWAY": ("HWY", "HIGHWAY"),
    "HIGHWY": ("HWY", "HIGHWAY"),
    "HIWAY": ("HWY", "HIGHWAY"),
    "HIWY": ("HWY", "HIGHWAY"),
    "HWAY": ("HWY", "HIGHWAY"),
    "HWY": ("HWY", "HIGHWAY"),
    "HILL": ("HL", "HILL"),
    "HL": ("HL", "HILL"),
    "HILLS": ("HLS", "HILLS"),
    "HLS": ("HLS", "HILLS"),
    "HLLW": ("HOLW", "HOLLOW"),
    "HOLLOW": ("HOLW", "HOLLOW"),
    "HOLLOWS": ("HOLW", "HOLLOW"),
    "HOLW": ("HOLW", "HOLLOW"),
    "HOLWS": ("HOLW", "HOLLOW"),
    "INLET": ("INLT", "INLET"),
    "INLT": ("INLT", "INLET"),
    "IS": ("IS", "ISLAND"),
    "ISLAND": ("IS", "ISLAND"),
    "ISLND": ("IS", "ISLAND"),
    "ISLANDS": ("ISS", "ISLANDS"),
    "ISLNDS": ("ISS", "ISLANDS"),
    "ISS": ("ISS", "ISLANDS"),
    "ISLE": ("ISLE", "ISLE"),
    "ISLES": ("ISLE", "ISLE"),
    "JCT": ("JCT", "JUNCTION"),
    "JCTION": ("JCT", "JUNCTION"),
    "JCTN": ("JCT", "JUNCTION"),
    "JUNCTION": ("JCT", "JUNCTION"),
    "JUNCTN": ("JCT", "JUNCTION"),
    "JUNCTON": ("JCT", "JUNCTION"),
    "JCTNS": ("JCTS", "JUNCTIONS"),
    "JCTS": ("JCTS", "JUNCTIONS"),
    "JUNCTIONS": ("JCTS", "JUNCTIONS"),
    "KEY": ("KY", "KEY"),
    "KY": ("KY", "KEY"),
    "KEYS": ("KYS", "KEYS"),
    "KYS": ("KYS", "KEYS"),
    "KNL": ("KNL", "KNOLL"),
    "KNOL": ("KNL", "KNOLL"),
    "KNOLL": ("KNL", "KNOLL"),
    "KNLS": ("KNLS", "KNOLLS"),
    "KNOLLS": ("KNLS", "KNOLLS"),
    "LAKE": ("LK", "LAKE"),
    "LK": ("LK", "LAKE"),
    "LAKES": ("LKS", "LAKES"),
    "LKS": ("LKS", "LAKES"),
    "LAND": ("LAND", "LAND"),
    "LANDING": ("LNDG", "LANDING"),
    "LNDG": ("LNDG", "LANDING"),
    "LNDNG": ("LNDG", "LANDING"),
    "LA": ("LN", "LANE"),
    "LANE": ("LN", "LANE"),
    "LANES": ("LN", "LANE"),
    "LN": ("LN", "LANE"),
    "LGT": ("LGT", "LIGHT"),
    "LIGHT": ("LGT", "LIGHT"),
    "LIGHTS": ("LGTS", "LIGHTS"),
    "LF": ("LF", "LOAF"),
    "LOAF": ("LF", "LOAF"),
    "LCK": ("LCK", "LOCK"),
    "LOCK": ("LCK", "LOCK"),
    "LCKS": ("LCKS", "LOCKS"),
    "LOCKS": ("LCKS", "LOCKS"),
    "LDG": ("LDG", "LODGE"),
    "LDGE": ("LDG", "LODGE"),
    "LODG": ("LDG", "LODGE"),
    "LODGE": ("LDG", "LODGE"),
    "LOOP": ("LOOP", "LOOP"),
    "LOOPS": ("LOOP", "LOOP"),
    "MALL": ("MALL", "MALL"),
    "MANOR": ("MNR", "MANOR"),
    "MNR": ("MNR", "MANOR"),
    "MANORS": ("MNRS", "MANORS"),
    "MNRS": ("MNRS", "MANORS"),
    "MDW": ("MDW", "MEADOW"),
    "MEADOW": ("MDW", "MEADOW"),
    "MDWS": ("MDWS", "MEADOWS"),
    "MEADOWS": ("MDWS", "MEADOWS"),
    "MEDOWS": ("MDWS", "MEADOWS"),
    "MEWS": ("MEWS", "MEWS"),
    "MILL": ("ML", "MILL"),
    "ML": ("ML", "MILL"),
    "MILLS": ("MLS", "MILLS"),
    "MLS": ("MLS", "MILLS"),
    "MISSION": ("MSN", "MISSION"),
    "MISSN": ("MSN", "MISSION"),
    "MSN": ("MSN", "MISSION"),
    "MSSN": ("MSN", "MISSION"),
    "MOTORWAY": ("MTWY", "MOTORWAY"),
    "MNT": ("MT", "MOUNT"),
    "MOUNT": ("MT", "MOUNT"),
    "MT": ("MT", "MOUNT"),
    "MNTAIN": ("MTN", "MOUNTAIN"),
    "MNTN": ("MTN", "MOUNTAIN"),
    "MOUNTAIN": ("MTN", "MOUNTAIN"),
    "MOUNTIN": ("MTN", "MOUNTAIN"),
    "MTIN": ("MTN", "MOUNTAIN"),
    "MTN": ("MTN", "MOUNTAIN"),
    "MNTNS": ("MTNS", "MOUNTAINS"),
    "MOUNTAINS": ("MTNS", "MOUNTAINS"),
    "NCK": ("NCK", "NECK"),
    "NECK": ("NCK", "NECK"),
    "ORCH": ("ORCH", "ORCHARD"),
    "ORCHARD": ("ORCH", "ORCHARD"),
    "ORCHRD": ("ORCH", "ORCHARD"),
    "OVAL": ("OVAL", "OVAL"),
    "OVL": ("OVAL", "OVAL"),
    "OVERPASS": ("OPAS", "OVERPASS"),
    "PARK": ("PARK", "PARK"),
    "PK": ("PARK", "PARK"),
    "PRK": ("PARK", "PARK"),
    "PARKS": ("PARK", "PARKS"),
    "PARKWAY": ("PKWY", "PARKWAY"),
    "PARKWY": ("PKWY", "PARKWAY"),
    "PKWAY": ("PKWY", "PARKWAY"),
    "PKWY": ("PKWY", "PARKWAY"),
    "PKY": ("PKWY", "PARKWAY"),
    "PARKWAYS": ("PKWY", "PARKWAY"),
    "PKWYS": ("PKWY", "PARKWAY"),
    "PASS": ("PASS", "PASS"),
    "PASSAGE": ("PSGE", "PASSAGE"),
    "PATH": ("PATH", "PATH"),
    "PATHS": ("PATH", "PATH"),
    "PIKE": ("PIKE", "PIKE"),
    "PIKES": ("PIKE", "PIKE"),
    "PINE": ("PNE", "PINE"),
    "PINES": ("PNES", "PINES"),
    "PNES": ("PNES", "PINES"),
    "PL": ("PL", "PLACE"),
    "PLACE": ("PL", "PLACE"),
    "PLAIN": ("PLN", "PLAIN"),
    "PLN": ("PLN", "PLAIN"),
    "PLAINES": ("PLNS", "PLAINS"),
    "PLAINS": ("PLNS", "PLAINS"),
    "PLNS": ("PLNS", "PLAINS"),
    "PLAZA": ("PLZ", "PLAZA"),
    "PLZ": ("PLZ", "PLAZA"),
    "PLZA": ("PLZ", "PLAZA"),
    "POINT": ("PT", "POINT"),
    "PT": ("PT", "POINT"),
    "POINTS": ("PTS", "POINTS"),
    "PTS": ("PTS", "POINTS"),
    "PORT": ("PRT", "PORT"),
    "PRT": ("PRT", "PORT"),
    "PORTS": ("PRTS", "PORTS"),
    "PRTS": ("PRTS", "PORTS"),
    "PR": ("PR", "PRAIRIE"),
    "PRAIRIE": ("PR", "PRAIRIE"),
    "PRARIE": ("PR", "PRAIRIE"),
    "PRR": ("PR", "PRAIRIE"),
    "RAD": ("RADL", "RADIAL"),
    "RADIAL": ("RADL", "RADIAL"),
    "RADIEL": ("RADL", "RADIAL"),
    "RADL": ("RADL", "RADIAL"),
    "RAMP": ("RAMP", "RAMP"),
    "RANCH": ("RNCH", "RANCH"),
    "RANCHES": ("RNCH", "RANCH"),
    "RNCH": ("RNCH", "RANCH"),
    "RNCHS": ("RNCH", "RANCH"),
    "RAPID": ("RPD", "RAPID"),
    "RPD": ("RPD", "RAPID"),
    "RAPIDS": ("RPDS", "RAPIDS"),
    "RPDS": ("RPDS", "RAPIDS"),
    "REST": ("RST", "REST"),
    "RST": ("RST", "REST"),
    "RDG": ("RDG", "RIDGE"),
    "RDGE": ("RDG", "RIDGE"),
    "RIDGE": ("RDG", "RIDGE"),
    "RDGS": ("RDGS", "RIDGES"),
    "RIDGES": ("RDGS", "RIDGES"),
    "RIV": ("RIV", "RIVER"),
    "RIVER": ("RIV", "RIVER"),
    "RIVR": ("RIV", "RIVER"),
    "RVR": ("RIV", "RIVER"),
    "RD": ("RD", "ROAD"),
    "ROAD": ("RD", "ROAD"),
    "RDS": ("RDS", "ROADS"),
    "ROADS": ("RDS", "ROADS"),
    "ROUTE": ("RTE", "ROUTE"),
    "ROW": ("ROW", "ROW"),
    "RUE": ("RUE", "RUE"),
    "RUN": ("RUN", "RUN"),
    "SHL": ("SHL", "SHOAL"),
    "SHOAL": ("SHL", "SHOAL"),
    "SHLS": ("SHLS", "SHOALS"),
    "SHOALS": ("SHLS", "SHOALS"),
    "SHOAR": ("SHR", "SHORE"),
    "SHORE": ("SHR", "SHORE"),
    "SHR": ("SHR", "SHORE"),
    "SHOARS": ("SHRS", "SHORES"),
    "SHORES": ("SHRS", "SHORES"),
    "SHRS": ("SHRS", "SHORES"),
    "SKYWAY": ("SKWY", "SKYWAY"),
    "SPG": ("SPG", "SPRING"),
    "SPNG": ("SPG", "SPRING"),
    "SPRING": ("SPG", "SPRING"),
    "SPRNG": ("SPG", "SPRING"),
    "SPGS": ("SPGS", "SPRINGS"),
    "SPNGS": ("SPGS", "SPRINGS"),
    "SPRINGS": ("SPGS", "SPRINGS"),
    "SPRNGS": ("SPGS", "SPRINGS"),
    "SPUR": ("SPUR", "SPUR"),
    "SPURS": ("SPUR", "SPUR"),
    "SQ": ("SQ", "SQUARE"),
    "SQR": ("SQ", "SQUARE"),
    "SQRE": ("SQ", "SQUARE"),
    "SQU": ("SQ", "SQUARE"),
    "SQUARE": ("SQ", "SQUARE"),
    "SQRS": ("SQS", "SQUARES"),
    "SQUARES": ("SQS", "SQUARES"),
    "STA": ("STA", "STATION"),
    "STATION": ("STA", "STATION"),
    "STATN": ("STA", "STATION"),
    "STN": ("STA", "STATION"),
    "STRA": ("STRA", "STRAVENUE"),
    "STRAV": ("STRA", "STRAVENUE"),
    "STRAVE": ("STRA", "STRAVENUE"),
    "STRAVEN": ("STRA", "STRAVENUE"),
    "STRAVENUE": ("STRA", "STRAVENUE"),
    "STRAVN": ("STRA", "STRAVENUE"),
    "STRVN": ("STRA", "STRAVENUE"),
    "STRVNUE": ("STRA", "STRAVENUE"),
    "STREAM": ("STRM", "STREAM"),
    "STREME": ("STRM", "STREAM"),
    "STRM": ("STRM", "STREAM"),
    "ST": ("ST", "STREET"),
    "STR": ("ST", "STREET"),
    "STREET": ("ST", "STREET"),
    "STRT": ("ST", "STREET"),
    "STREETS": ("STS", "STREETS"),
    "SMT": ("SMT", "SUMMIT"),
    "SUMIT": ("SMT", "SUMMIT"),
    "SUMITT": ("SMT", "SUMMIT"),
    "SUMMIT": ("SMT", "SUMMIT"),
    "TER": ("TER", "TERRACE"),
    "TERR": ("TER", "TERRACE"),
    "TERRACE": ("TER", "TERRACE"),
    "THROUGHWAY": ("TRWY", "THROUGHWAY"),
    "TRACE": ("TRCE", "TRACE"),
    "TRACES": ("TRCE", "TRACE"),
    "TRCE": ("TRCE", "TRACE"),
    "TRACK": ("TRAK", "TRACK"),
    "TRACKS": ("TRAK", "TRACK"),
    "TRAK": ("TRAK", "TRACK"),
    "TRK": ("TRAK", "TRACK"),
    "TRKS": ("TRAK", "TRACK"),
    "TRAFFICWAY": ("TRFY", "TRAFFICWAY"),
    "TRFY": ("TRFY", "TRAFFICWAY"),
    "TR": ("TRL", "TRAIL"),
    "TRAIL": ("TRL", "TRAIL"),
    "TRAILS": ("TRL", "TRAIL"),
    "TRL": ("TRL", "TRAIL"),
    "TRLS": ("TRL", "TRAIL"),
    "TUNEL": ("TUNL", "TUNNEL"),
    "TUNL": ("TUNL", "TUNNEL"),
    "TUNLS": ("TUNL", "TUNNEL"),
    "TUNNEL": ("TUNL", "TUNNEL"),
    "TUNNELS": ("TUNL", "TUNNEL"),
    "TUNNL": ("TUNL", "TUNNEL"),
    "TPK": ("TPKE", "TURNPIKE"),
    "TPKE": ("TPKE", "TURNPIKE"),
    "TRNPK": ("TPKE", "TURNPIKE"),
    "TRPK": ("TPKE", "TURNPIKE"),
    "TURNPIKE": ("TPKE", "TURNPIKE"),
    "TURNPK": ("TPKE", "TURNPIKE"),
    "UNDERPASS": ("UPAS", "UNDERPASS"),
    "UN": ("UN", "UNION"),
    "UNION": ("UN", "UNION"),
    "UNIONS": ("UNS", "UNIONS"),
    "VALLEY": ("VLY", "VALLEY"),
    "VALLY": ("VLY", "VALLEY"),
    "VLLY": ("VLY", "VALLEY"),
    "VLY": ("VLY", "VALLEY"),
    "VALLEYS": ("VLYS", "VALLEYS"),
    "VLYS": ("VLYS", "VALLEYS"),
    "VDCT": ("VIA", "VIADUCT"),
    "VIA": ("VIA", "VIADUCT"),
    "VIADCT": ("VIA", "VIADUCT"),
    "VIADUCT": ("VIA", "VIADUCT"),
    "VIEW": ("VW", "VIEW"),
    "VW": ("VW", "VIEW"),
    "VIEWS": ("VWS", "VIEWS"),
    "VWS": ("VWS", "VIEWS"),
    "VILL": ("VLG", "VILLAGE"),
    "VILLAG": ("VLG", "VILLAGE"),
    "VILLAGE": ("VLG", "VILLAGE"),
    "VILLG": ("VLG", "VILLAGE"),
    "VILLIAGE": ("VLG", "VILLAGE"),
    "VLG": ("VLG", "VILLAGE"),
    "VILLAGES": ("VLGS", "VILLAGES"),
    "VLGS": ("VLGS", "VILLAGES"),
    "VILLE": ("VL", "VILLE"),
    "VL": ("VL", "VILLE"),
    "VIS": ("VIS", "VISTA"),
    "VIST": ("VIS", "VISTA"),
    "VISTA": ("VIS", "VISTA"),
    "VST": ("VIS", "VISTA"),
    "VSTA": ("VIS", "VISTA"),
    "WALK": ("WALK", "WALK"),
    "WALKS": ("WALKS", "WALKS"),
    "WALL": ("WALL", "WALL"),
    "WAY": ("WAY", "WAY"),
    "WY": ("WAY", "WAY"),
    "WAYS": ("WAYS", "WAYS"),
    "WELL": ("WL", "WELL"),
    "WELLS": ("WLS", "WELLS"),
    "WLS": ("WLS", "WELLS"),
}

# The address components are based upon the `United States Thoroughfare,
# Landmark, and Postal Address Data Standard
# http://www.urisa.org/advocacy/united-states-thoroughfare-landmark-and-postal-address-data-standard

LABELS = [
    "AddressNumberPrefix",
    "AddressNumber",
    "AddressNumberSuffix",
    "StreetNamePreModifier",
    "StreetNamePreDirectional",
    "StreetNamePreType",
    "StreetName",
    "StreetNamePostType",
    "StreetNamePostDirectional",
    "SubaddressType",
    "SubaddressIdentifier",
    "BuildingName",
    "OccupancyType",
    "OccupancyIdentifier",
    "CornerOf",
    "LandmarkName",
    "PlaceName",
    "StateName",
    "ZipCode",
    "USPSBoxType",
    "USPSBoxID",
    "USPSBoxGroupType",
    "USPSBoxGroupID",
    "IntersectionSeparator",
    "Recipient",
    "NotAddress",
]

PARENT_LABEL = "AddressString"
GROUP_LABEL = "AddressCollection"

MODEL_FILE = "usaddr.crfsuite"
MODEL_PATH = os.path.split(os.path.abspath(__file__))[0] + "/" + MODEL_FILE

DIRECTIONS = set(
    [
        "n",
        "s",
        "e",
        "w",
        "ne",
        "nw",
        "se",
        "sw",
        "north",
        "south",
        "east",
        "west",
        "northeast",
        "northwest",
        "southeast",
        "southwest",
    ]
)

STREET_NAMES = {
    "allee",
    "alley",
    "ally",
    "aly",
    "anex",
    "annex",
    "annx",
    "anx",
    "arc",
    "arcade",
    "av",
    "ave",
    "aven",
    "avenu",
    "avenue",
    "avn",
    "avnue",
    "bayoo",
    "bayou",
    "bch",
    "beach",
    "bend",
    "bg",
    "bgs",
    "bl",
    "blf",
    "blfs",
    "bluf",
    "bluff",
    "bluffs",
    "blvd",
    "bnd",
    "bot",
    "bottm",
    "bottom",
    "boul",
    "boulevard",
    "boulv",
    "br",
    "branch",
    "brdge",
    "brg",
    "bridge",
    "brk",
    "brks",
    "brnch",
    "brook",
    "brooks",
    "btm",
    "burg",
    "burgs",
    "byp",
    "bypa",
    "bypas",
    "bypass",
    "byps",
    "byu",
    "camp",
    "canyn",
    "canyon",
    "cape",
    "causeway",
    "causwa",
    "causway",
    "cen",
    "cent",
    "center",
    "centers",
    "centr",
    "centre",
    "ci",
    "cir",
    "circ",
    "circl",
    "circle",
    "circles",
    "cirs",
    "ck",
    "clb",
    "clf",
    "clfs",
    "cliff",
    "cliffs",
    "club",
    "cmn",
    "cmns",
    "cmp",
    "cnter",
    "cntr",
    "cnyn",
    "common",
    "commons",
    "cor",
    "corner",
    "corners",
    "cors",
    "course",
    "court",
    "courts",
    "cove",
    "coves",
    "cp",
    "cpe",
    "cr",
    "crcl",
    "crcle",
    "crecent",
    "creek",
    "cres",
    "crescent",
    "cresent",
    "crest",
    "crk",
    "crossing",
    "crossroad",
    "crossroads",
    "crscnt",
    "crse",
    "crsent",
    "crsnt",
    "crssing",
    "crssng",
    "crst",
    "crt",
    "cswy",
    "ct",
    "ctr",
    "ctrs",
    "cts",
    "curv",
    "curve",
    "cv",
    "cvs",
    "cyn",
    "dale",
    "dam",
    "div",
    "divide",
    "dl",
    "dm",
    "dr",
    "driv",
    "drive",
    "drives",
    "drs",
    "drv",
    "dv",
    "dvd",
    "est",
    "estate",
    "estates",
    "ests",
    "ex",
    "exp",
    "expr",
    "express",
    "expressway",
    "expw",
    "expy",
    "ext",
    "extension",
    "extensions",
    "extn",
    "extnsn",
    "exts",
    "fall",
    "falls",
    "ferry",
    "field",
    "fields",
    "flat",
    "flats",
    "fld",
    "flds",
    "fls",
    "flt",
    "flts",
    "ford",
    "fords",
    "forest",
    "forests",
    "forg",
    "forge",
    "forges",
    "fork",
    "forks",
    "fort",
    "frd",
    "frds",
    "freeway",
    "freewy",
    "frg",
    "frgs",
    "frk",
    "frks",
    "frry",
    "frst",
    "frt",
    "frway",
    "frwy",
    "fry",
    "ft",
    "fwy",
    "garden",
    "gardens",
    "gardn",
    "gateway",
    "gatewy",
    "gatway",
    "gdn",
    "gdns",
    "glen",
    "glens",
    "gln",
    "glns",
    "grden",
    "grdn",
    "grdns",
    "green",
    "greens",
    "grn",
    "grns",
    "grov",
    "grove",
    "groves",
    "grv",
    "grvs",
    "gtway",
    "gtwy",
    "harb",
    "harbor",
    "harbors",
    "harbr",
    "haven",
    "havn",
    "hbr",
    "hbrs",
    "height",
    "heights",
    "hgts",
    "highway",
    "highwy",
    "hill",
    "hills",
    "hiway",
    "hiwy",
    "hl",
    "hllw",
    "hls",
    "hollow",
    "hollows",
    "holw",
    "holws",
    "hrbor",
    "ht",
    "hts",
    "hvn",
    "hway",
    "hwy",
    "inlet",
    "inlt",
    "is",
    "island",
    "islands",
    "isle",
    "isles",
    "islnd",
    "islnds",
    "iss",
    "jct",
    "jction",
    "jctn",
    "jctns",
    "jcts",
    "junction",
    "junctions",
    "junctn",
    "juncton",
    "key",
    "keys",
    "knl",
    "knls",
    "knol",
    "knoll",
    "knolls",
    "ky",
    "kys",
    "la",
    "lake",
    "lakes",
    "land",
    "landing",
    "lane",
    "lanes",
    "lck",
    "lcks",
    "ldg",
    "ldge",
    "lf",
    "lgt",
    "lgts",
    "light",
    "lights",
    "lk",
    "lks",
    "ln",
    "lndg",
    "lndng",
    "loaf",
    "lock",
    "locks",
    "lodg",
    "lodge",
    "loop",
    "loops",
    "lp",
    "mall",
    "manor",
    "manors",
    "mdw",
    "mdws",
    "meadow",
    "meadows",
    "medows",
    "mews",
    "mi",
    "mile",
    "mill",
    "mills",
    "mission",
    "missn",
    "ml",
    "mls",
    "mn",
    "mnr",
    "mnrs",
    "mnt",
    "mntain",
    "mntn",
    "mntns",
    "motorway",
    "mount",
    "mountain",
    "mountains",
    "mountin",
    "msn",
    "mssn",
    "mt",
    "mtin",
    "mtn",
    "mtns",
    "mtwy",
    "nck",
    "neck",
    "opas",
    "orch",
    "orchard",
    "orchrd",
    "oval",
    "overlook",
    "overpass",
    "ovl",
    "ovlk",
    "park",
    "parks",
    "parkway",
    "parkways",
    "parkwy",
    "pass",
    "passage",
    "path",
    "paths",
    "pike",
    "pikes",
    "pine",
    "pines",
    "pk",
    "pkway",
    "pkwy",
    "pkwys",
    "pky",
    "pl",
    "place",
    "plain",
    "plaines",
    "plains",
    "plaza",
    "pln",
    "plns",
    "plz",
    "plza",
    "pne",
    "pnes",
    "point",
    "points",
    "port",
    "ports",
    "pr",
    "prairie",
    "prarie",
    "prk",
    "prr",
    "prt",
    "prts",
    "psge",
    "pt",
    "pts",
    "pw",
    "pwy",
    "rad",
    "radial",
    "radiel",
    "radl",
    "ramp",
    "ranch",
    "ranches",
    "rapid",
    "rapids",
    "rd",
    "rdg",
    "rdge",
    "rdgs",
    "rds",
    "rest",
    "ri",
    "ridge",
    "ridges",
    "rise",
    "riv",
    "river",
    "rivr",
    "rn",
    "rnch",
    "rnchs",
    "road",
    "roads",
    "route",
    "row",
    "rpd",
    "rpds",
    "rst",
    "rte",
    "rue",
    "run",
    "rvr",
    "shl",
    "shls",
    "shoal",
    "shoals",
    "shoar",
    "shoars",
    "shore",
    "shores",
    "shr",
    "shrs",
    "skwy",
    "skyway",
    "smt",
    "spg",
    "spgs",
    "spng",
    "spngs",
    "spring",
    "springs",
    "sprng",
    "sprngs",
    "spur",
    "spurs",
    "sq",
    "sqr",
    "sqre",
    "sqrs",
    "sqs",
    "squ",
    "square",
    "squares",
    "st",
    "sta",
    "station",
    "statn",
    "stn",
    "str",
    "stra",
    "strav",
    "strave",
    "straven",
    "stravenue",
    "stravn",
    "stream",
    "street",
    "streets",
    "streme",
    "strm",
    "strt",
    "strvn",
    "strvnue",
    "sts",
    "sumit",
    "sumitt",
    "summit",
    "te",
    "ter",
    "terr",
    "terrace",
    "throughway",
    "tl",
    "tpk",
    "tpke",
    "tr",
    "trace",
    "traces",
    "track",
    "tracks",
    "trafficway",
    "trail",
    "trailer",
    "trails",
    "trak",
    "trce",
    "trfy",
    "trk",
    "trks",
    "trl",
    "trlr",
    "trlrs",
    "trls",
    "trnpk",
    "trpk",
    "trwy",
    "tunel",
    "tunl",
    "tunls",
    "tunnel",
    "tunnels",
    "tunnl",
    "turn",
    "turnpike",
    "turnpk",
    "un",
    "underpass",
    "union",
    "unions",
    "uns",
    "upas",
    "valley",
    "valleys",
    "vally",
    "vdct",
    "via",
    "viadct",
    "viaduct",
    "view",
    "views",
    "vill",
    "villag",
    "village",
    "villages",
    "ville",
    "villg",
    "villiage",
    "vis",
    "vist",
    "vista",
    "vl",
    "vlg",
    "vlgs",
    "vlly",
    "vly",
    "vlys",
    "vst",
    "vsta",
    "vw",
    "vws",
    "walk",
    "walks",
    "wall",
    "way",
    "ways",
    "well",
    "wells",
    "wl",
    "wls",
    "wy",
    "xc",
    "xg",
    "xing",
    "xrd",
    "xrds",
}


def load_model() -> Any:
    """
    Load address parsing model from local directory.
    """
    tagger = None
    try:
        tagger = pycrfsuite.Tagger()
        tagger.open(MODEL_PATH)
    except IOError:
        warnings.warn(
            "You must train the model (parserator train --trainfile "
            "FILES) to create the %s file before you can use the parse "
            "and tag methods" % MODEL_FILE
        )
    return tagger


def parse(address_string: str) -> List[Any]:
    """
    Function to parse address.

    Parameters
    ----------
    address_string
        The address string with flexible format.
    """

    tagger = load_model()

    tokens = tokenize(address_string)

    if not tokens:
        return []

    features = tokens2features(tokens)

    tags = tagger.tag(features)
    return list(zip(tokens, tags))


def tag(
    address_string: str, tag_mapping: Optional[Dict[str, str]] = None
) -> Tuple[Dict[str, Any], str]:
    """
    Function to tag each part of address.

    Parameters
    ----------
    address_string
        The address string with flexible format.
    tag_mapping
        The dictionary used for mapping each part of address
    """
    tagged_address: Dict[str, Any] = OrderedDict()

    last_label = None
    is_intersection = False
    og_labels = []

    for token, label in parse(address_string):
        if label == "IntersectionSeparator":
            is_intersection = True
        if "StreetName" in label and is_intersection:
            label = "Second" + label

        # saving old label
        og_labels.append(label)
        # map tag to a new tag if tag mapping is provided
        if tag_mapping and tag_mapping.get(label):
            label = tag_mapping.get(label)

        if label == last_label:
            tagged_address[label].append(token)
        elif label not in tagged_address:
            tagged_address[label] = [token]
        else:
            raise RepeatedLabelError(address_string, parse(address_string), label)

        last_label = label

    for token in tagged_address:
        component = " ".join(tagged_address[token])
        component = component.strip(" ,;")
        tagged_address[token] = component

    if "AddressNumber" in og_labels and not is_intersection:
        address_type = "Street Address"
    elif is_intersection and "AddressNumber" not in og_labels:
        address_type = "Intersection"
    elif "USPSBoxID" in og_labels:
        address_type = "PO Box"
    else:
        address_type = "Ambiguous"

    return tagged_address, address_type


def tokenize(address_string: str) -> Any:
    """
    Function to tokenize address.

    Parameters
    ----------
    address_string
        The address string with flexible format.
    """
    if isinstance(address_string, bytes):
        address_string = str(address_string, encoding="utf-8")
    address_string = re.sub("(&#38;)|(&amp;)", "&", address_string)
    re_tokens = re.compile(
        r"""
    \(*\b[^\s,;#&()]+[.,;)\n]*   # ['ab. cd,ef '] -> ['ab.', 'cd,', 'ef']
    |
    [#&]                       # [^'#abc'] -> ['#']
    """,
        re.VERBOSE | re.UNICODE,
    )

    tokens = re_tokens.findall(address_string)

    if not tokens:
        return []

    return tokens


def transform_token_features(token: str) -> Any:
    """
    Function to extract feature for each token.

    Parameters
    ----------
    token
        The string of token.
    """
    if token in ("&", "#", "½"):
        token_clean = token
    else:
        token_clean = re.sub(r"(^[\W]*)|([^.\w]*$)", "", token, flags=re.UNICODE)

    token_abbrev = re.sub(r"[.]", "", token_clean.lower())
    features = {
        "abbrev": token_clean[-1] == ".",
        "digits": digits(token_clean),
        "word": (token_abbrev if not token_abbrev.isdigit() else False),
        "trailing.zeros": (trailing_zeros(token_abbrev) if token_abbrev.isdigit() else False),
        "length": (
            "d:" + str(len(token_abbrev))
            if token_abbrev.isdigit()
            else "w:" + str(len(token_abbrev))
        ),
        "endsinpunc": (
            token[-1] if bool(re.match(r".+[^.\w]", token, flags=re.UNICODE)) else False
        ),
        "directional": token_abbrev in DIRECTIONS,
        "street_name": token_abbrev in STREET_NAMES,
        "has.vowels": bool(set(token_abbrev[1:]) & set("aeiou")),
    }

    return features


def tokens2features(address: Any) -> Any:
    """
    Function to convert token sequence to features.

    Parameters
    ----------
    address
        Tokens composing address.
    """
    feature_sequence = [transform_token_features(address[0])]
    previous_features = feature_sequence[-1].copy()

    for token in address[1:]:
        token_features = transform_token_features(token)
        current_features = token_features.copy()

        feature_sequence[-1]["next"] = current_features
        token_features["previous"] = previous_features

        feature_sequence.append(token_features)

        previous_features = current_features

    feature_sequence[0]["address.start"] = True
    feature_sequence[-1]["address.end"] = True

    if len(feature_sequence) > 1:
        feature_sequence[1]["previous"]["address.start"] = True
        feature_sequence[-2]["next"]["address.end"] = True

    return feature_sequence


def digits(token: str) -> str:
    """
    Function to judge if the current token is digital.

    Parameters
    ----------
    token
        The token string.
    """
    if token.isdigit():
        return "all_digits"
    elif set(token) & set(string.digits):
        return "some_digits"
    else:
        return "no_digits"


def trailing_zeros(token: str) -> Any:
    """
    Function all zeros in the current token.

    Parameters
    ----------
    token
        The token string.
    """
    results = re.findall(r"(0+)$", token)
    if results:
        return results[0]
    else:
        return ""


class RepeatedLabelError(Exception):
    """Repeated label error report.
    Attributes:
        REPO_URL: URL of usaddress repo.
        DOCS_URL: URL of usaddress documentation.
        MESSAGE: Error message.
        DOC_MESSAGE: Documentation message.
        message: Output message
        original_string: The string of original data
        parsed_string: The string of parsed data
    """

    REPO_URL = "https://github.com/datamade/usaddress/issues/new"
    DOCS_URL = "https://usaddress.readthedocs.io/"

    MESSAGE = """
ERROR: Unable to tag this string because more than one area of the string has the same label
ORIGINAL STRING:  {original_string}
PARSED TOKENS:    {parsed_string}
UNCERTAIN LABEL:  {repeated_label}
When this error is raised, it's likely that either (1) the string is not a valid person/corporation name or (2) some tokens were labeled incorrectly
To report an error in labeling a valid name, open an issue at {repo_url} - it'll help us continue to improve probablepeople!"""

    DOCS_MESSAGE = """
For more information, see the documentation at {docs_url}"""

    def __init__(
        self, original_string: str, parsed_string: List[Any], repeated_label: Optional[str]
    ) -> None:
        """
        This function initiates the RepeatedLabelError.

        Parameters
        ----------
        original_string
            The string of original data.
        parsed_string
            The string of parsed data.
        repeated_label
            The strings of repeated labels.
        """
        Exception.__init__(self)
        self.message = self.MESSAGE.format(
            original_string=original_string,
            parsed_string=parsed_string,
            repeated_label=repeated_label,
            repo_url=self.REPO_URL,
        )
        if self.DOCS_URL:
            self.message += self.DOCS_MESSAGE.format(docs_url=self.DOCS_URL)

        self.original_string = original_string
        self.parsed_string = parsed_string

    def __str__(self) -> Any:
        """
        This function returns the message from the RepeatedLabelError.
        """
        return self.message
