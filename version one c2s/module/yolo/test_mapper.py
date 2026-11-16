import os
import glob
import shutil
import yaml

def test_mapper():
    # Canonical classes
    CANONICAL_CLASSES = {
        0: 'plastic',
        1: 'glass',
        2: 'metal',
        3: 'paper',
        4: 'cardboard',
        5: 'organic',
        6: 'hazardous'
    }

    # Keyword mapping
    def map_class_name(name):
        n = str(name).lower().strip().replace('_', ' ').replace('-', ' ')
        if any(k in n for k in ['plastic', 'bottle', 'container', 'jug', 'canister', 'polythene', 'bag', 'wrap', 'folder']):
            return 0 # plastic
        if any(k in n for k in ['glass', 'jar', 'stemware', 'wine glass', 'cup']):
            return 1 # glass
        if any(k in n for k in ['metal', 'can', 'tin', 'aluminium', 'aluminum', 'foil', 'drinkcan', 'foodcan']):
            return 2 # metal
        if any(k in n for k in ['paper', 'newspaper', 'receipt', 'paperbag', 'tissue', 'book', 'drinkpack']):
            return 3 # paper
        if any(k in n for k in ['cardboard', 'carton', 'box']):
            return 4 # cardboard
        if any(k in n for k in ['banana', 'chilli', 'lettuce', 'potato', 'food', 'fruit', 'veggie', 'tea', 'organic', 'vegetable', 'scrap']):
            return 5 # organic
        if any(k in n for k in ['battery', 'electronic', 'phone', 'pcb', 'remote', 'wire', 'e waste', 'hazardous']):
            return 6 # hazardous
        return 0 # default to plastic / general

    print('plasticbottle ->', map_class_name('plasticbottle'))
    print('drinkcan      ->', map_class_name('drinkcan'))
    print('foodcan       ->', map_class_name('foodcan'))
    print('banana        ->', map_class_name('banana'))
    print('glass         ->', map_class_name('glass'))
    print('cardboard     ->', map_class_name('cardboard'))
    print('paperbag      ->', map_class_name('paperbag'))

if __name__ == '__main__':
    test_mapper()
