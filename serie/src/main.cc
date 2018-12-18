int main(int argc, char* argv[]) {
    if (argc != 2) {
        cout << "usage: ./" << argv[0] << " csv_filename";
        return 1;
    }

    string filename(argv[1]);
    Dataset dataset(filename);

    Model model();
    model.train(dataset);

    return 0;
}
