




# defining type k2.Array


class Array:

    # `indexes` is a Tensor with one
    Tensor indexes;

    # `data` is either:
    #   - of type Tensor (if this corresponds to Array2 == 2-dimensional
    #     array in C++)
    #   - of type Array (if this corresponds to Array3 or higher-dimensional
    #     array in C++)
    # The Python code is structured a bit differently from the C++ code,
    # due to the differences in the languages.
    # When we dispatch things to C++ code there would be some
    # big switch statement or if-statement to select the right
    # template instantiation.
    data;

    def __len__(self):
        return indexes.shape[0] - 1

    @property
    def shape(self):
        # e.g. if indexes.shape is (15,) and
        # data.shape is (150) -> this.shape would be (15,None)
        # If data.shape is (150,4), this.shape would be (15,4)
        # If data.shape is (150,None) (since data is an Array), this.shape
        #     would be (150,None,None).
        # The Nones are for dimensions where the shape is not known
        # because it is variable.
        return (indexes.shape[0] - 1, None, *data.shape[1:])



class Fsa(Array):

    # Think of this as a vector of vector of Arc, or in C++,
    # an Array2<Arc>.
    # An Arc has 3 int32_t's, so this.data is a Tensor with
    # dtype int32 and shape (_, 3).



class FsaVec(Array):

    # Think of this as a vector of vector of vector of Arc, or in C++,
    # an Array3<Arc>.
    #
    # this.data is an Array, and this.data.data is a Tensor with
    # dtype int32 and shape (_, 3).
