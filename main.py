import click
import nervestitcher
import fusion
import config


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--threshold",
    "-t",
    default=config.SCORE_THRESHOLD,
    help="interestpoint score threshold (superpoint)",
)
@click.option("--preprocess", "-p", is_flag=True, default=True, help="do preprocessing step")
@click.option("--images", "-i", "image_path")
@click.option("--output", "-o", "output_path")
def extract_interestpoints(image_path, output_path, preprocess, threshold):
    images = nervestitcher.load_images_in_directory(image_path)
    if preprocess:
        images = nervestitcher.preprocess_images(images)
    data = fusion.generate_interest_point_data(images, threshold)
    fusion.save_interest_point_data(output_path, *data)


@cli.command()
@click.option("--diagonals", default=0, help="how many superdiagonals to match per image")
@click.option(
    "--threshold", default=config.MATCHING_THRESHOLD, help="matching threshold (superglue)"
)
@click.option("--images", "-i", "image_path")
@click.option("--output", "-o", "output_path")
@click.option("--interest-point-data", "-d", "interest_point_path")
def extract_match_data(image_path, interest_point_path, output_path, diagonals, threshold):
    images = nervestitcher.load_images_in_directory(image_path)
    data = fusion.load_interest_point_data(interest_point_path)
    match_data = fusion.generate_raw_match_matrix(
        images, *data, diagonals=diagonals, threshold=threshold
    )
    fusion.save_raw_match_matrix(output_path, match_data)


@cli.command()
@click.option("--images", "-i", "image_path")
@click.option("--output", "-o", "output_path")
@click.option("--match-data", "-d", "match_data_path")
@click.option("--preprocess", "-p", is_flag=True, default=True, help="do preprocessing step")
def fuse(image_path, output_path, match_data_path, preprocess):
    images = nervestitcher.load_images_in_directory(image_path)
    if preprocess:
        images = nervestitcher.preprocess_images(images)
    raw_match_matrix = fusion.load_raw_match_matrix(match_data_path)
    match_translation_matrix = fusion.get_match_translation_matrix_from_raw_match_matrix(
        raw_match_matrix
    )
    height, width = images[0].shape
    positions_x, positions_y = fusion.solve_match_translation_matrix(
        match_translation_matrix, width, height
    )
    fusion.fuse(images, positions_x, positions_y)


@cli.command()
def gui():
    # TODO
    pass


cli()
