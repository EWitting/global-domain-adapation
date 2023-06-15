import os

from util.batch import batch_generate
from experiment.presets.bias import concept_weak, concept_strong, covariate_weak, covariate_strong

PREFIX = "v4"


def gen(builder, num, name):
    batch_path = os.path.join(os.getcwd(), '../results', PREFIX, name)
    batch_generate(builder, num, batch_path)


for bias in [concept_weak,
             concept_strong,
             covariate_weak,
             covariate_strong]:
    gen(bias(), 5, bias.__name__)
    gen(bias(), 10, f"{bias.__name__}_val")


