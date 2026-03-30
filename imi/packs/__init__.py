"""IMI Memory Packs — pre-built domain knowledge.

Usage:
    from imi.packs import load_sre_pack
    space = load_sre_pack("my_sre.db")
    # Space comes pre-loaded with 500 SRE incidents
    result = space.navigate("DNS outage")
"""
