"""Count clusters, surname forms and positive pairs in the Gaois Sloinnte minimal.xml.

Usage: python count_pairs.py [path/to/minimal.xml]
"""
import sys
import xml.etree.ElementTree as ET
from itertools import combinations

IRISH = ["surnames-irish-main", "surnames-irish-historical", "surnames-irish-alternative"]
ENGLISH = ["surnames-english-main", "surnames-english-historical", "surnames-english-alternative"]


def get_forms(cluster, sections):
    """Return the set of distinct base forms in the given sections of one cluster."""
    forms = set()
    for section in sections:
        for form in cluster.findall(f"./{section}//form"):
            if form.text:
                forms.add(form.text.strip())
    return forms


def main(path):
    clusters = ET.parse(path).getroot().findall(".//cluster")

    n_irish = n_english = 0
    irish_irish = irish_english = 0
    unique_pairs = set()  # to check for pairs repeated across clusters

    for cluster in clusters:
        irish = get_forms(cluster, IRISH)
        english = get_forms(cluster, ENGLISH)
        n_irish += len(irish)
        n_english += len(english)

        # Irish-Irish synonyms: every unordered pair of distinct Irish forms
        for a, b in combinations(sorted(irish), 2):
            irish_irish += 1
            unique_pairs.add((a, b))

        # Irish-English equivalents: every Irish form with every English form
        for a in irish:
            for b in english:
                irish_english += 1
                unique_pairs.add((a, b))

    print(f"Clusters:             {len(clusters)}")
    print(f"Irish forms:          {n_irish}")
    print(f"English forms:        {n_english}")
    print(f"Irish-Irish pairs:    {irish_irish}")
    print(f"Irish-English pairs:  {irish_english}")
    print(f"Total positive pairs: {irish_irish + irish_english}")
    print(f"Unique positive pairs (duplicates across clusters removed): {len(unique_pairs)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "minimal.xml")