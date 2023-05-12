import sys
import os
from specxplore import importing

def test_empty() -> None:
    assert True
    return None

# server side downtime and the like affect the testing of the server access
#def test_get_classes():
    """ Get classification using inchi key. Compare results to previous 09-05-2023 result for consistency. """
    #inchi1 = "InChI=1S/C15H10O6/c16-8-3-1-7(2-4-8)15-14(20)13(19)12-10(18)5-9(17)6-11(12)21-15/h1-6,16-18,20H"
    #inchi2 = "InChI=1S/C21H20O11/c22-6-14-17(28)18(29)19(30)21(32-14)16-11(26)4-10(25)15-12(27)5-13(31-20(15)16)7-1-2-8(23)9(24)3-7/h1-5,14,17-19,21-26,28-30H,6H2/t14-,17-,18+,19-,21+/m1/s1"
    #classes_validated1 = importing.ClassificationEntry(
    #    inchi='InChI=1S/C15H10O6/c16-8-3-1-7(2-4-8)15-14(20)13(19)12-10(18)5-9(17)6-11(12)21-15/h1-6,16-18,20H',
    #    smiles='O%3Dc1c%28O%29c%28-c2ccc%28O%29cc2%29oc2cc%28O%29cc%28O%29c12',
    #    cf_kingdom='Organic compounds',  cf_superclass='Phenylpropanoids and polyketides', 
    #    cf_class='Flavonoids', cf_subclass='Flavones', cf_direct_parent='Flavonols', npc_class='Flavonols', 
    #    npc_superclass='Flavonoids', npc_pathway='Shikimates and Phenylpropanoids', npc_isglycoside='0')
    #classes1 = importing.get_classes(inchi1)
    #classes_validated2 = importing.ClassificationEntry(
    #    inchi='InChI=1S/C21H20O11/c22-6-14-17(28)18(29)19(30)21(32-14)16-11(26)4-10(25)15-12(27)5-13(31-20(15)16)7-1-2-8(23)9(24)3-7/h1-5,14,17-19,21-26,28-30H,6H2/t14-,17-,18+,19-,21+/m1/s1', 
    #    smiles='O%3Dc1cc%28-c2ccc%28O%29c%28O%29c2%29oc2c%28%5BC%40%40H%5D3O%5BC%40H%5D%28CO%29%5BC%40%40H%5D%28O%29%5BC%40H%5D%28O%29%5BC%40H%5D3O%29c%28O%29cc%28O%29c12', 
    #    cf_kingdom='Organic compounds', cf_superclass='Phenylpropanoids and polyketides', cf_class='Flavonoids', 
    #    cf_subclass='Flavonoid glycosides', cf_direct_parent='Flavonoid 8-C-glycosides', npc_class='Flavones',
    #    npc_superclass='Flavonoids', npc_pathway='Shikimates and Phenylpropanoids', npc_isglycoside='0')
    #classes2 = importing.get_classes(inchi2)
    #assert classes1 == classes_validated1, "Classification entry does not match expected output of 09-05-2023"
    #assert classes2 == classes_validated2, "Classification entry does not match expected output of 09-05-2023"

#if __name__ == "__main__":
#    test_get_classes() # gnps api call, lengthy step