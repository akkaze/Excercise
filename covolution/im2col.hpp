void im2col(double* im,
	double* col,
	const int& nums,
	const int& channels,
	const int& rows,
	const int& cols,
	const int& filter_h,
	const int& filter_w,
	const int& padding,
	const int& stride)
{
	const int& new_rows = ( 
		rows + 2 * padding - filter_h) /
		stride + 1;
	const int& new_cols = ( 
		rows + 2 * padding - filter_h) /
		stride + 1;

	for(int hi = 0; hi < filter_h; hi++)
		for(int wi = 0; wi < filter_w; wi++)
			for(int ci = 0; ci < channels; ci++)
			{
				const int& r = hi * 
					filter_w * 
					channels + 
					wi * 
					channels + ci;
				for(int i = 0; i < nums; i++)
					for(int hj = 0; hj < new_h; hj++)
						for(int wj = 0; wj < new_w; wj++)
						{
							const int& c = i *
								new_h *
								new_w +
								hj *
								new_w + wj;
							const int& orig_row = stride * 
										hj + 
										hi - padding;
							const int& orig_col = stride * 
										wj + 
										wi - padding;
							if(orig_row >=0 && orig_row < rows &&
								orig_col >=0 && orig_col < cols)
							{
								*(col + r * channels *
								 filter_h * 
								 filter_w + c)= 
								*(im + i * rows * 
									cols * 
									channels +
									orig_row * 
									cols *
									channels +
									orig_col *
									channels +
									ci);
							}
							else
								*(col + r * channels *
								 filter_h * 
								 filter_w + c)= 0;
						}
			}
}



void col2im(double* col,
	double* im,
	const int& nums,
	const int& channels,
	const int& rows,
	const int& cols,
	const int& filter_h,
	const int& filter_w,
	const int& padding,
	const int& stride)
{
	const int& new_rows = ( 
		rows + 2 * padding - filter_h) /
		stride + 1;
	const int& new_cols = ( 
		rows + 2 * padding - filter_h) /
		stride + 1;

	for(int hi = 0; hi < filter_h; hi++)
		for(int wi = 0; wi < filter_w; wi++)
			for(int ci = 0; ci < channels; ci++)
			{
				const int& r = hi * 
					filter_w * 
					channels + 
					wi * 
					channels + ci;
				for(int i = 0; i < nums; i++)
					for(int hj = 0; hj < new_h; hj++)
						for(int wj = 0; wj < new_w; wj++)
						{
							const int& c = i *
								new_h *
								new_w +
								hj *
								new_w + wj;
							const int& orig_row = stride * 
										hj + 
										hi - padding;
							const int& orig_col = stride * 
										wj + 
										wi - padding;
							if(orig_row >=0 && orig_row < rows &&
								orig_col >=0 && orig_col < cols)
							{
								*(im + i * rows * 
									cols * 
									channels +
									orig_row * 
									cols *
									channels +
									orig_col *
									channels +
									ci) = 
									*(col + r * channels *
								 	filter_h * 
								 	filter_w + c)= 
								
							}
						}
			}

}