/**
 * @file common.cpp
 * @author Krystof Teissing (k.teissing@gmail.com)
 * @version 0.1
 * @date 2023-05-25
 * 
 */

#include "common.hpp"

namespace pmm {

void operator>>(const YAML::Node& node, Scalar& value) {
  assert(node.IsScalar());
  value = node.as<Scalar>();
}

void operator>>(const YAML::Node& node, Vector<3>& value) {
    std::vector<Scalar> tmp;
    assert(node.IsSequence());
    tmp = node.as<std::vector<Scalar>>();
    value[0] = tmp[0];
    value[1] = tmp[1];
    value[2] = tmp[2];
}

} // namespace pmm