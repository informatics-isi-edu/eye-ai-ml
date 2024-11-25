# Tests for the datapath module.
#
# Environment variables:
#  DERIVA_PY_TEST_HOSTNAME: hostname of the test server
#  DERIVA_PY_TEST_CREDENTIAL: user credential, if none, it will attempt to get credentail for given hostname
#  DERIVA_PY_TEST_VERBOSE: set for verbose logging output to stdout
import logging
import os
import sys
import unittest

import pandas as pd
from deriva.core import DerivaServer, ErmrestCatalog, get_credential
from typing import Optional

from deriva_ml.deriva_ml_base import DerivaML, DerivaMLException, RID, ColumnDefinition, BuiltinTypes
from deriva_ml.schema_setup.create_schema import create_ml_schema
from eye_ai.schema_setup.test_catalog import create_domain_schema, populate_test_catalog
from eye_ai.eye_ai import EyeAI
from deriva_ml.execution_configuration import ExecutionConfiguration

try:
    from pandas import DataFrame

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

SNAME_DOMAIN = 'eye-ai'

hostname = os.getenv("DERIVA_PY_TEST_HOSTNAME")
logger = logging.getLogger(__name__)
if os.getenv("DERIVA_PY_TEST_VERBOSE"):
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())


test_catalog: Optional[ErmrestCatalog] = None


def setUpModule():
    global test_catalog
    logger.debug("setUpModule begin")
    credential = os.getenv("DERIVA_PY_TEST_CREDENTIAL") or get_credential(hostname)
    server = DerivaServer('https', hostname, credentials=credential)
    test_catalog = server.create_ermrest_catalog()
    model = test_catalog.getCatalogModel()
    try:
        create_ml_schema(model)
        create_domain_schema(model, SNAME_DOMAIN)
        populate_test_catalog(model, SNAME_DOMAIN)
    except Exception:
        # on failure, delete catalog and re-raise exception
        test_catalog.delete_ermrest_catalog(really=True)
        raise
    logger.debug("setUpModule  done")


def tearDownModule():
    logger.debug("tearDownModule begin")
    test_catalog.delete_ermrest_catalog(really=True)
    logger.debug("tearDownModule done")


@unittest.skipUnless(hostname, "Test host not specified")
class TestEyeAI(unittest.TestCase):

    def setUp(self):
        self.ml_instance = EyeAI(hostname, test_catalog.catalog_id, SNAME_DOMAIN, None, None, "1")
        self.domain_schema = self.ml_instance.model.schemas[SNAME_DOMAIN]
        self.model = self.ml_instance.model

    def tearDown(self):
        pass

    def test_insert_new_diagnosis(self):
        def insert_new_diagnosis(self, pred_df: pd.DataFrame,
                                 diagtag_rid: str,
                                 execution_rid: str):
        populate_test_catalog(self.model, SNAME_DOMAIN)
        pred_df = pd.DataFrame()
        diagtag_rid = self.ml_instance.lookup_term("Diagnosis_Tag")
        execution_rid = self.ml_instance.lookup_term("Execution_RID")
        self.ml_instance.insert_new_diagnosis(pred_df, diagtag_rid, execution_rid)
        self.assertIn("Dataset_Type", [v.name for v in self.ml_instance.find_vocabularies()])

    def test_create_vocabulary(self):
        populate_test_catalog(self.model, SNAME_DOMAIN)
        self.ml_instance.create_vocabulary("CV1", "A vocab")




if __name__ == '__main__':
    sys.exit(unittest.main())
