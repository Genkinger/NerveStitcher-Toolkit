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
@click.option("--preprocess", "-p", is_flag=True, help="do preprocessing step")
@click.option("--images", "-i", "image_path", required=True)
@click.option("--output", "-o", "output_path", required=True)
def extract_interest_points(image_path, output_path, preprocess, threshold):
    images = nervestitcher.load_images_in_directory(image_path)
    if preprocess:
        images = nervestitcher.preprocess_images(images)
    data = fusion.generate_interest_point_data(images, threshold)
    fusion.save_interest_point_data(output_path, data)


@cli.command()
@click.option("--diagonals", default=0, help="how many superdiagonals to match per image")
@click.option(
    "--threshold", default=config.MATCHING_THRESHOLD, help="matching threshold (superglue)"
)
# @click.option("--images", "-i", "image_path", required=True)
@click.option("--output", "-o", "output_path", required=True)
@click.option("--interest-point-data", "-d", "interest_point_path", required=True)
def extract_match_data(interest_point_path, output_path, diagonals, threshold):
    data = fusion.load_interest_point_data(interest_point_path)
    match_data = fusion.generate_raw_match_data(data, diagonals=diagonals, threshold=threshold)
    fusion.save_raw_match_matrix(output_path, match_data)


@cli.command()
@click.option("--images", "-i", "image_path", required=True)
@click.option("--output", "-o", "output_path", required=True)
@click.option("--match-data", "-d", "match_data_path", required=True)
@click.option("--preprocess", "-p", is_flag=True, default=True, help="do preprocessing step")
def fuse(image_path, output_path, match_data_path, preprocess):
    # images = nervestitcher.load_images_in_directory(image_path)
    # if preprocess:
    #     images = nervestitcher.preprocess_images(images)
    # raw_match_matrix = fusion.load_raw_match_matrix(match_data_path)
    # match_translation_matrix = fusion.get_match_translation_matrix_from_raw_match_matrix(
    #     raw_match_matrix
    # )
    # height, width = images[0].shape
    # positions_x, positions_y = fusion.solve_match_translation_matrix(
    #     match_translation_matrix, width, height
    # )
    # fusion.fuse(images, positions_x, positions_y)
    pass


@cli.command()
def gui():
    # TODO
    pass


cli()
