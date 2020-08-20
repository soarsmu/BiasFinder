import en_core_web_lg
import neuralcoref
nlp = en_core_web_lg.load()
coref = neuralcoref.NeuralCoref(nlp.vocab)
nlp.add_pipe(coref, name='neuralcoref')