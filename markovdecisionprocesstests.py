import unittest
import markovdecisionprocess as mdp_module


class MarkovDecisionProcessTests(unittest.TestCase):

    def test_init(self):
        state_count = 5
        mdp = mdp_module.MarkovDecisionProcess(state_count)
        self.assertIsNotNone(mdp)
        self.assertEqual(state_count, mdp.states_count)
        self.assertEqual((state_count, ), mdp.states.shape)


if __name__ == '__main__':
    unittest.main()
