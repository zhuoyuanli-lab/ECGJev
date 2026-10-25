def detached_view(features):
    return features.detach() if features is not None else None
