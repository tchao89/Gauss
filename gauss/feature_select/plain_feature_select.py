features_enc, feature_names_enc = ft.selection.remove_low_information_features(features, feature_names)
features_enc, feature_names_enc = ft.selection.remove_highly_correlated_features(features_enc,
                                                                                 feature_names_enc)
features_enc, feature_names_enc = ft.selection.remove_highly_null_features(features_enc, feature_names_enc)
features_enc, feature_names_enc = ft.selection.remove_single_value_features(features_enc, feature_names_enc)