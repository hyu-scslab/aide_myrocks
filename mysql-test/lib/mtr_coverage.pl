use strict;

# Extract coverage option
#
# Arguments:
#  $option       coverage option
#  $delim        delimiter for splitting the string
#  $option_error error to be printed in case of failure
sub coverage_extract_option {
  my ($option, $delim, $option_error) = @_;
  # check for sanity of option which should be of the format --option=value
  if (length($option) == 0) {
      print "**** ERROR **** ",
            "Invalid coverage option specified for $option_error\n";
      exit(1);
  }

  # split the string on delimiter '='
  my @option_arr = split(/$delim/, $option);
  $option=$option_arr[$#option_arr];

  return $option;
}

# Prepare to generate coverage data
#
# Arguments:
#   $build_dir  build directory
#   $base_dir   basedir, normally the home directory
#   $scope      coverage option which is of the form:
#                 * full : complete code coverage
#                 * diff : coverage of the git diff HEAD
#                 * diff:<commit_hash> : coverage of git diff commit_hash
#   $src_path   directory path for coverage source files
#   $llvm_path  directory for llvm coverage binaries
#   $format     format for coverage report which is of the form:
#                 * text : text format
#                 * html : html format
#   $src_filter src filter directories
sub coverage_prepare($$$) {
  my ($build_dir, $base_dir, $scope, $src_path, $llvm_path, $format,
      $src_filter) = @_;

  print "Purging coverage information from '$base_dir'...\n";
  system("find $base_dir -name \"code\*.profraw\" | xargs rm");

  my $scope = coverage_extract_option($scope, "=", "coverage-scope");

  my $commit_hash = "HEAD"; # default commit hash is 'HEAD'
  # if the coverage scope is "--full" then extract the git commithash
  if ($scope =~ m/^diff/) {
      my $invalid_commit_hash = 0; # is this commit hash valid?
      # if the coverage scope is of the form 'diff:<commit_hash>'
      if ($scope =~ /^diff:/) {
          $commit_hash = coverage_extract_option($scope, ":",
                                                 "coverage-scope");
          # sanity check for commit hash
          if (length($commit_hash) == 0) {
              $invalid_commit_hash = 1;
          }
      }
      # if the coverage scope is of the form '--diff'
      elsif ($scope ne "diff") {
          $invalid_commit_hash = 1;
      }

      if ($invalid_commit_hash) {
          print "**** ERROR **** ",
                "Invalid coverage scope diff option: $scope\n";
          exit(1);
      }
  }
  # make sure that the coverage scope is "--full"
  elsif ($scope ne "full") {
    print "**** ERROR **** ", "Invalid coverage scope: $scope\n";
    exit(1);
  }

  # Update the scope of the coverage
  if ($scope eq "full") {
    $_[2] = $scope;
  }
  else {
    $_[2] = $commit_hash;
  }

  # extract directory for coverage source files
  $src_path = coverage_extract_option($src_path, "=", "coverage-src-path")."/";

  # Update the coverage src path
  $_[3] = $src_path;

  $llvm_path = coverage_extract_option($llvm_path, "=", "coverage-llvm-path");

  # append "/" at the end of the llvm_path
  if (length($llvm_path) > 0 && ! ($llvm_path =~ m/\/$/) ) {
    $llvm_path .= "/";
  }

  # Update the coverage llvm path
  $_[4] = $llvm_path;

  # extract format for coverage report
  $format = coverage_extract_option($format, "=", "coverage-format");

  # sanity check for coverage format
  if ( ! ( ($format eq "text") || ($format eq "html")) ) {
      print "**** ERROR **** ", "Invalid coverage-format option: $format\n";
      exit(1);
  }

  $_[5] = $format;

  # process "--coverage-src-filter" arguments
  $src_filter =~ s/^\s+//g;
  my @filter_arr = split(/ /, $src_filter);
  my $final_src_filter="";
  my $f;
  foreach $f (@filter_arr) {
    my $filter_path = coverage_extract_option($f, "=", "coverage-src-filter");
    if (length($final_src_filter) > 0) {
        $final_src_filter .= ",";
    }
    $final_src_filter .= $filter_path;
  }

  # Update the coverage src filter
  $_[6] = $final_src_filter;

  # create the directory to store the generated coverage files
  my $mkdir_cmd = "$build_dir/coverage_files";
  mkpath($mkdir_cmd);
}

# Get the files modified by a git diff
#
# Arguments:
#  $src_dir      directory for coverage source files
#  $commit_hash  git commit hash
sub coverage_get_diff_files ($$) {
  my ($src_dir, $commit_hash) = @_;

  # command to extract files modified by a git commit hash
  my $cmd = "git diff --name-only $commit_hash"."^ $commit_hash";
  open(PIPE, "$cmd|");

  my $commit_hash_files; # concatenated list of files

  while(<PIPE>) {
      chomp;
      if (/\.h$/ or /\.cc$/) {
          $commit_hash_files .= $src_dir.$_." ";
      }
  }
  return $commit_hash_files;
}

# Merge coverage profile files
#
# Arguments:
#   $merge_com   command for merging the coverage profile files
#   $results_dir directory that contains coverage results
sub coverage_merge_prof_files($$) {
  my ($merge_com, $result_dir) = @_;

  my $err_file = "$result_dir/err";
  my $merge_com_err = "$merge_com 2>$err_file";
  # If the merge command fails due to corrupted profile files
  # then repeat merging the files by deleting the corrupt files
  while (1) {
    system($merge_com_err);
    open(FP, "<", $err_file) or dir $!;
    my $got_error = 0; # did we see an error
    my $line;
    while($line = <FP>) {
      $got_error = 1;
      chomp($line);
      my @files_list = split(":", $line);
      my $corrupt_file = $files_list[1];
      $corrupt_file =~ s/ //g;
      # delete the corrupt file
      rmtree("$corrupt_file");
    }
    # there is no error, bail out
    if ($got_error == 0) {
      last;
    }
  }
}

# Collect coverage information
#
# Arguments:
#  $build_dir   build directory
#  $base_dir    basedir, normally the home directory
#  $binary_path path to mysqld binary
#  $scope       coverage option which is of the form:
#                 * full : complete code coverage
#                 * HEAD : coverage of the git diff HEAD
#                 * <commit_hash> : coverage of git diff commit_hash
#  $src_path    directory path for coverage source files
#  $llvm_path   directory for llvm coverage binaries
#  $format      format for coverage report which is of the form:
#                 * text : text format
#                 * html : html format
#  $src_filter  src filter directories
#  $cov_comand  coverage command issued (used for logging)
sub coverage_collect ($$$) {
  my ($build_dir, $base_dir, $binary_path, $scope, $src_path, $llvm_path,
      $format, $src_filter, $cov_command) = @_;

  my $files_modified=""; # list of files modified concatenated into one string

  if ($scope ne "full") {
    $files_modified = coverage_get_diff_files($src_path, $scope);
  }

  print "Generating coverage information ";
  if (length($files_modified) > 0) {
    print "for git commit hash $scope";
    if ($scope eq "HEAD") {
      # command to extract git commit hash of 'HEAD'
      my $cmd = "git rev-parse $scope";
      open(PIPE, "$cmd|");
      my $head_commit_hash = <PIPE>;
      chomp($head_commit_hash);
      print " ($head_commit_hash)";
    }
  }
  elsif (length($src_filter) > 0) {
    my @dir_paths = $src_filter =~ /,/g;
    my $num_dirs = @dir_paths+1;
    my $dir_str = "directory";
    if ($num_dirs > 1) {
        $dir_str = "directories";
    }
    print "for source files in $dir_str: $src_filter";
  }
  else {
    print "for complete source code ";
  }
  print " ...\n";

  # Create directory to store the coverage results
  my $result_dir = "$build_dir/reports/".time();
  mkpath($result_dir);

  # Recreate the 'last' directory to point to the latest coverage
  # results directory
  my $last_dir = "$build_dir/reports/last";
  rmtree($last_dir);
  symlink($result_dir, $last_dir);

  # Log the command used for generating coverage in command.log
  open(FP, ">$result_dir/command.log");
  print FP "$cov_command\n";
  close(FP);

  # Merge coverage reports using command
  # llvm-prof merge --output=file.profdata <list of code*.profraw>
  my $merge_cov = $llvm_path."llvm-profdata merge --output=";
  $merge_cov .= $result_dir."/combined.profdata ";
  $merge_cov .= "`find $build_dir -name \"code*.profraw\"`";
  coverage_merge_prof_files($merge_cov, $result_dir);

  # Generate coverage report using command
  # llvm-cov show <binary_path> --instr-profile=file.profdata \
  #      --format <text|html> --output-dir=<output_dir>
  my $generate_cov
      = $llvm_path."llvm-cov show $binary_path -instr-profile=";
  $generate_cov .= $result_dir."/combined.profdata ";

  # Add the list of files modified if the coverage report is for
  # a specific git diff
  if (length($files_modified) > 0) {
      $generate_cov .= $files_modified;
  } elsif (length($src_filter) > 0) {
    # if files modified is empty then apply the coverage-src-filter
    my @filter_paths = split(/,/, $src_filter);
    my $single_filter;
    foreach $single_filter (@filter_paths) {
      my $find_cmd
          = "`find $src_path/$single_filter -name \"*.cc\" -o -name \"*.h\"";
      $find_cmd .= " -o -name \"*.c\" -o -name \"*.ic\"`";
      $generate_cov .= " $find_cmd ";
    }
  }

  $generate_cov .= "--format $format"; # coverage report format
  $generate_cov .= " --output-dir=$result_dir 2>/dev/null";
  system($generate_cov);

  # Delete profdata file
  print "Purging coverage related files...\n";
  my $rm_profdata = $result_dir."/combined.profdata";
  rmtree($rm_profdata);
  my $rm_coverage_files = "$build_dir/coverage_files";
  rmtree($rm_coverage_files);
  system("find $base_dir -name \"code\*.profraw\" | xargs rm");

  print "Completed generating coverage information in ",
        "$format format.\n";
  print "Coverage results directory: $build_dir/reports/last\n";
}

1;
