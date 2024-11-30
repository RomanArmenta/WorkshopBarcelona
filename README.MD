# SCHNETJAX

Here, I try to implement the SchNet architecture in JAX/FLAX from scratch, and train a little database of water molecules.

- Firstly, the model does not converge. I think it has to do with the filter generator or the convolutional layer, so I need to check that part very closely. ---- I already revise it all. I put relu function in case it was the problem, but no. I must try to do it by blocks.

--------------

TODO

- Try to do it by blocks.

- Generate filter plots.

- Add initializers (normal) to the embeddings.

- Try with a simpler NN (FFNN, conv, etc.).