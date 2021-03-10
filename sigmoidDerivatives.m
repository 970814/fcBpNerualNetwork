function sd = sigmoidDerivatives(z)
    a = sigmoid(z);
    sd = a  .* (1 - a);
end;