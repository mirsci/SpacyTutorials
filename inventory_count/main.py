import inventory_count.inventoryCount as mainModule
import os
import spacy
# Very important, as the models are loaded in a different location
spacy.util.set_data_path('/data/projects/python3.4/spacy')

from spacy.en import English

if __name__ == '__main__':
    """
    Main module for this example - loads the English main NLP class,
    and keeps it in RAM while waiting for the user to re-run it. Allows the
    developer to re-edit their module under testing without having
    to wait as long to load the English class
    """

    #  Set the NLP object here for the parameters you want to see,
    #  or just leave it blank and get all the opts
    print ("Loading English module... this will take a while.")
    nlp = English()
    print ("Done loading English module.")
    while True:
        try:
            # mainModule.reload()
            mainModule.runTest(nlp)
            input('================ To reload main module, press Enter ================')


        except Exception as e:
            print ("Unexpected error: ", str(e))
            continue