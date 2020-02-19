from ncaa_predict.data.access import mens_access, womens_access
from ncaa_predict.features import registry


def main():
    features_results = registry.run(access=womens_access)
    features_results = registry.run(access=womens_access)
    print(features_results)


if __name__ == '__main__':
    main()
